import random
import numpy as np
import torch

from .constants import BASE, PRECISION_FRACTIONAL, Q
from .crypto import aes_ctr_prg
from .datasets import row_normalize
from .dice import generate_dg_props, DiceEnsembleSampler
from .dot_products import dot_product_matrix_rss_optimized_batch, dot_product_matrix_rss_batch_gpu
from .noise_sampling import get_dp_noise_shares
from .rss import (
    encode,
    decode,
    share_vector,
    share_vector_positive,
    reconstruct_batch,
    boolean_share_vector_batch,
    inverse_permutation_batch,
    compose_perm_batch,
    semi_shuffle_batch,
    B2A_batch,
)
from .verification import Protocol2Verifier

class BasePreTrainer:
    def __init__(self, data, args, features, client_node_indices):
        self.data = data
        self.features = features.cpu().numpy()
        self.args = args
        self.device = args["device"]
        self.client_node_indices = client_node_indices
        self.N = self.data.num_nodes
        self.F_dim = self.features.shape[1]
        self.batch_size = args["batch_size"]
        self.adj_list = {i: [] for i in range(self.N)}
        cpu_edge_index = self.data.edge_index.cpu().numpy()
        for src, dst in zip(cpu_edge_index[0], cpu_edge_index[1]):
            self.adj_list[src].append(dst)
        for i in range(self.N):
            if i not in self.adj_list[i]:
                self.adj_list[i].append(i)
        self.d_max = max(len(neighbors) for neighbors in self.adj_list.values())

    def build_permutation_batch(self, start_idx, end_idx):
        batch_size = end_idx - start_idx
        a_prime_comp_batch = np.zeros((batch_size, self.d_max), dtype=np.int64)
        pi_batch = np.zeros((batch_size, self.N), dtype=int)
        for b_idx, node_i in enumerate(range(start_idx, end_idx)):
            neighbors = self.adj_list[node_i]
            k_i = len(neighbors)
            a_prime_comp_batch[b_idx, :k_i] = 1
            L_i = np.array(neighbors, dtype=int)
            E_i = np.setdiff1d(np.arange(self.N), L_i)
            pi_batch[b_idx, :k_i] = L_i
            pi_batch[b_idx, k_i:] = E_i
        return a_prime_comp_batch, pi_batch

    def complete_shuffle_permutations(self, pi_batch):
        batch_size = pi_batch.shape[0]
        pi0_batch = np.random.rand(batch_size, self.N).argsort(axis=1)
        pi1_batch = np.random.rand(batch_size, self.N).argsort(axis=1)
        inv_pi0_batch = inverse_permutation_batch(pi0_batch)
        inv_pi1_batch = inverse_permutation_batch(pi1_batch)
        temp = compose_perm_batch(inv_pi0_batch, pi_batch)
        pi2_batch = compose_perm_batch(inv_pi1_batch, temp)
        P_batch = pi2_batch[:, :self.d_max]
        pi2_prime_batch = np.zeros((batch_size, self.N), dtype=int)
        pi2_prime_batch[:, :self.d_max] = P_batch
        for b_idx in range(batch_size):
            rem_indices = np.setdiff1d(np.arange(self.N), P_batch[b_idx])
            pi2_prime_batch[b_idx, self.d_max:] = rem_indices
        return [pi2_prime_batch, pi1_batch, pi0_batch]

    def boolean_masks(self, batch_size):
        masks = [np.random.randint(0, 2, (batch_size, self.N, 3), dtype=np.int64) for _ in range(3)]
        for mask in masks:
            mask[:, :, 2] = mask[:, :, 0] ^ mask[:, :, 1]
        return masks

    def arithmetic_masks(self, batch_size):
        masks = [np.random.randint(0, Q, (batch_size, self.N, 3), dtype=np.int64) for _ in range(3)]
        for mask in masks:
            mask[:, :, 2] = (-mask[:, :, 0] - mask[:, :, 1]) % Q
        return masks

    def padded_boolean_adjacency(self, a_prime_comp_bool_batch, batch_size):
        pad_shares = np.zeros((batch_size, self.N - self.d_max, 3), dtype=np.int64)
        return np.concatenate([a_prime_comp_bool_batch, pad_shares], axis=1)

class SemiHonestPreTrainer(BasePreTrainer):
    def perform_secure_aggregation(self):
        current_X = row_normalize(torch.tensor(self.features), eps=self.args["eps"]).numpy()
        X_shares = share_vector(encode(current_X))
        features_list = [current_X]
        for hop in range(self.args["hops"]):
            new_X_shares = np.zeros_like(X_shares)
            for start_idx in range(0, self.N, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.N)
                batch_size = end_idx - start_idx
                a_prime_comp_batch, pi_batch = self.build_permutation_batch(start_idx, end_idx)
                a_prime_comp_bool_batch = boolean_share_vector_batch(a_prime_comp_batch)
                pi_list_batch = self.complete_shuffle_permutations(pi_batch)
                a_prime_bool_batch = self.padded_boolean_adjacency(a_prime_comp_bool_batch, batch_size)
                a_bool_shares_batch = semi_shuffle_batch(pi_list_batch, a_prime_bool_batch, self.boolean_masks(batch_size), is_boolean=True)
                a_arith_shares_batch = B2A_batch(a_bool_shares_batch)
                node_feat_shares_batch = dot_product_matrix_rss_optimized_batch(a_arith_shares_batch, X_shares)
                new_X_shares[start_idx:end_idx] = node_feat_shares_batch
            X_new_encoded = reconstruct_batch(new_X_shares)
            X_new_plain = np.vectorize(decode)(X_new_encoded)
            X_new_normalized = row_normalize(torch.tensor(X_new_plain, dtype=torch.float32), eps=self.args["eps"]).numpy()
            features_list.append(X_new_normalized)
            if hop < self.args["hops"] - 1:
                X_shares = share_vector(encode(X_new_normalized))
        final_features = np.concatenate(features_list, axis=1)
        return torch.tensor(final_features, dtype=torch.float32, device=self.device)

class DPPreTrainer(BasePreTrainer):
    def __init__(self, data, args, features, client_node_indices):
        super().__init__(data, args, features, client_node_indices)
        if self.args["dp_sigma"] > 0:
            sigma_scaled = self.args["dp_sigma"] * (BASE ** PRECISION_FRACTIONAL)
            target_pmf, support_vals = generate_dg_props(sigma_scaled, r=int(sigma_scaled * 20), L=int(sigma_scaled * 10), device=self.device)
            table_size = max(32, 2 * len(target_pmf))
            sampler = DiceEnsembleSampler(target_pmf, support_vals, table_size, device=self.device)
            sampler.build_tables(32)
            self.dp_tables = sampler.tables.cpu().numpy()
            self.dp_support = support_vals.cpu().numpy()

    def perform_secure_aggregation(self):
        current_X = row_normalize(torch.tensor(self.features), eps=self.args["eps"]).numpy()
        X_shares = share_vector_positive(encode(current_X))
        features_list = [current_X]
        for hop in range(self.args["hops"]):
            new_X_shares = np.zeros_like(X_shares)
            X_shares_gpu = torch.tensor(X_shares, dtype=torch.int64, device=self.device)
            for start_idx in range(0, self.N, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.N)
                batch_size = end_idx - start_idx
                a_prime_comp_batch, pi_batch = self.build_permutation_batch(start_idx, end_idx)
                a_prime_comp_bool_batch = boolean_share_vector_batch(a_prime_comp_batch)
                pi_list_batch = self.complete_shuffle_permutations(pi_batch)
                a_prime_bool_batch = self.padded_boolean_adjacency(a_prime_comp_bool_batch, batch_size)
                a_bool_shares_batch = semi_shuffle_batch(pi_list_batch, a_prime_bool_batch, self.boolean_masks(batch_size), is_boolean=True)
                a_arith_shares_batch = B2A_batch(a_bool_shares_batch)
                node_feat_shares_batch = dot_product_matrix_rss_batch_gpu(a_arith_shares_batch, X_shares_gpu, self.device, Q)
                if self.args["dp_sigma"] > 0:
                    noise_shares_flat = get_dp_noise_shares(
                        n_total=batch_size * self.F_dim,
                        tables_np=self.dp_tables,
                        support_vals_np=self.dp_support,
                        q=Q,
                        scale_factor=BASE**PRECISION_FRACTIONAL,
                        chunk_size=512,
                    )
                    node_feat_shares_batch = (node_feat_shares_batch + noise_shares_flat.reshape(batch_size, self.F_dim, 3)) % Q
                new_X_shares[start_idx:end_idx] = node_feat_shares_batch
            X_new_encoded = reconstruct_batch(new_X_shares)
            X_new_plain = np.vectorize(decode)(X_new_encoded)
            X_new_normalized = row_normalize(torch.tensor(X_new_plain, dtype=torch.float32), eps=self.args["eps"]).numpy()
            features_list.append(X_new_normalized)
            if hop < self.args["hops"] - 1:
                X_shares = share_vector_positive(encode(X_new_normalized))
        return torch.tensor(np.concatenate(features_list, axis=1), dtype=torch.float32, device=self.device)

class MaliciousPreTrainer(BasePreTrainer):
    def __init__(self, data, args, features, client_node_indices):
        super().__init__(data, args, features, client_node_indices)
        self.verifier = Protocol2Verifier()

    def perform_secure_aggregation(self):
        current_X = row_normalize(torch.tensor(self.features), eps=self.args["eps"]).numpy()
        X_shares = share_vector(encode(current_X))
        features_list = [current_X]
        for hop in range(self.args["hops"]):
            new_X_shares = np.zeros_like(X_shares)
            X_shares_gpu = torch.tensor(X_shares, dtype=torch.int64, device=self.device)
            for start_idx in range(0, self.N, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.N)
                batch_size = end_idx - start_idx
                a_prime_comp_batch, pi_batch = self.build_permutation_batch(start_idx, end_idx)
                k_seeds = [[random.getrandbits(128).to_bytes(16, "big") for _ in range(3)] for _ in range(batch_size)]
                k_shares_raw = np.zeros((batch_size, self.N, 3), dtype=np.int64)
                for b in range(batch_size):
                    k_shares_raw[b, :, 0] = aes_ctr_prg(k_seeds[b][0], 0, self.N)
                    k_shares_raw[b, :, 1] = aes_ctr_prg(k_seeds[b][1], 0, self.N)
                    k_shares_raw[b, :, 2] = aes_ctr_prg(k_seeds[b][2], 0, self.N)
                k_plain_batch = reconstruct_batch(k_shares_raw)
                t_plain_batch = np.zeros(batch_size, dtype=np.int64)
                for b_idx, node_i in enumerate(range(start_idx, end_idx)):
                    for j in range(len(self.adj_list[node_i])):
                        t_plain_batch[b_idx] = (t_plain_batch[b_idx] + int(k_plain_batch[b_idx, j])) % Q
                a_prime_comp_bool_batch = boolean_share_vector_batch(a_prime_comp_batch)
                t_shares_batch = share_vector(t_plain_batch)
                pi_list_batch = self.complete_shuffle_permutations(pi_batch)
                a_prime_bool_batch = self.padded_boolean_adjacency(a_prime_comp_bool_batch, batch_size)
                a_bool_shuffled = semi_shuffle_batch(pi_list_batch, a_prime_bool_batch, self.boolean_masks(batch_size), is_boolean=True)
                k_arith_shuffled = semi_shuffle_batch(pi_list_batch, k_shares_raw, self.arithmetic_masks(batch_size), is_boolean=False)
                a_arith_shuffled = B2A_batch(a_bool_shuffled)
                k_gpu = torch.tensor(k_arith_shuffled, device=self.device).unsqueeze(1)
                a_gpu_vec = torch.tensor(a_arith_shuffled, device=self.device).unsqueeze(2)
                t_prime_shares_np = dot_product_matrix_rss_batch_gpu(k_gpu, a_gpu_vec, self.device, Q)
                t_prime_shares = torch.tensor(t_prime_shares_np, device=self.device).view(batch_size, 3)
                delta = (t_prime_shares - torch.tensor(t_shares_batch, device=self.device)) % Q
                delta_plain = (delta[:, 0] + delta[:, 1] + delta[:, 2]) % Q
                if torch.any(delta_plain != 0):
                    raise RuntimeError(f"Integrity check failed at batch {start_idx}-{end_idx}.")
                Z_shares_batch = dot_product_matrix_rss_batch_gpu(a_arith_shuffled, X_shares_gpu, self.device, Q)
                if not self.verify_feature_aggregation(a_arith_shuffled, X_shares_gpu, Z_shares_batch):
                    raise RuntimeError("Feature aggregation verification failed.")
                new_X_shares[start_idx:end_idx] = Z_shares_batch
            X_new_encoded = reconstruct_batch(new_X_shares)
            X_new_plain = np.vectorize(decode)(X_new_encoded)
            X_new_normalized = row_normalize(torch.tensor(X_new_plain, dtype=torch.float32), eps=self.args["eps"]).numpy()
            features_list.append(X_new_normalized)
            if hop < self.args["hops"] - 1:
                X_shares = share_vector(encode(X_new_normalized))
        return torch.tensor(np.concatenate(features_list, axis=1), dtype=torch.float32, device=self.device)

    def verify_feature_aggregation(self, a_arith_shuffled, X_shares_gpu, Z_shares_batch):
        batch_size, node_count, _ = a_arith_shuffled.shape
        feature_dim = X_shares_gpu.shape[1]
        r1_np = np.random.randint(0, Q, (feature_dim, 1), dtype=np.int64).astype(object)
        r2_np = np.random.randint(0, Q, (batch_size, 1), dtype=np.int64).astype(object)
        X_cpu = X_shares_gpu.cpu().numpy().astype(object).transpose(2, 0, 1)
        v_cpu = np.array([np.dot(X_cpu[i], r1_np) % Q for i in range(3)])
        v_cpu = v_cpu.transpose(1, 2, 0).squeeze(1)
        A_cpu = a_arith_shuffled.astype(object).transpose(2, 0, 1)
        u_cpu = np.array([np.dot(r2_np.T, A_cpu[i]) % Q for i in range(3)])
        u_cpu = u_cpu.transpose(1, 2, 0).squeeze(0)
        Z_cpu = Z_shares_batch.astype(object).transpose(2, 0, 1)
        s_temp = np.array([np.dot(Z_cpu[i], r1_np) % Q for i in range(3)])
        s_cpu = np.array([np.dot(r2_np.T, s_temp[i]) % Q for i in range(3)])
        s_cpu = s_cpu.transpose(1, 2, 0).squeeze(0).reshape(1, 3)
        next_pow2 = 1 << (node_count - 1).bit_length()
        if next_pow2 > node_count:
            zeros = np.zeros((next_pow2 - node_count, 3), dtype=object)
            u_cpu = np.concatenate([u_cpu, zeros], axis=0)
            v_cpu = np.concatenate([v_cpu, zeros], axis=0)
        return self.verifier.verify_dot_product(u_cpu, v_cpu, s_cpu)

