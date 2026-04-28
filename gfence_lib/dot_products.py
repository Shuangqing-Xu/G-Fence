import numpy as np
import torch

from .constants import Q

def dot_mod_Q_batch(A, B, q=Q):
    shift = 21
    mask = (1 << shift) - 1
    A0 = A & mask
    A1 = (A >> shift) & mask
    A2 = A >> (2 * shift)
    B0 = B & mask
    B1 = (B >> shift) & mask
    B2 = B >> (2 * shift)
    D00 = np.dot(A0, B0)
    D01 = np.dot(A0, B1)
    D02 = np.dot(A0, B2)
    D10 = np.dot(A1, B0)
    D11 = np.dot(A1, B1)
    D12 = np.dot(A1, B2)
    D20 = np.dot(A2, B0)
    D21 = np.dot(A2, B1)
    D22 = np.dot(A2, B2)
    S0 = D00
    S1 = D01 + D10
    S2 = D02 + D11 + D20
    S3 = D12 + D21
    S4 = D22
    C1 = (1 << shift) % q
    C2 = (1 << (2 * shift)) % q
    C3 = (1 << (3 * shift)) % q
    C4 = (1 << (4 * shift)) % q
    Z = (S0.astype(object) +
         S1.astype(object) * C1 +
         S2.astype(object) * C2 +
         S3.astype(object) * C3 +
         S4.astype(object) * C4) % q
    return Z.astype(np.int64)

def dot_product_matrix_rss_optimized_batch(a_shares_batch, X_shares, q=Q):
    a = a_shares_batch.astype(np.int64)
    X = X_shares.astype(np.int64)
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    X0, X1, X2 = X[..., 0], X[..., 1], X[..., 2]
    z0 = (dot_mod_Q_batch(a0, X0, q) + dot_mod_Q_batch(a0, X1, q) + dot_mod_Q_batch(a1, X0, q)) % q
    z1 = (dot_mod_Q_batch(a1, X1, q) + dot_mod_Q_batch(a1, X2, q) + dot_mod_Q_batch(a2, X1, q)) % q
    z2 = (dot_mod_Q_batch(a2, X2, q) + dot_mod_Q_batch(a2, X0, q) + dot_mod_Q_batch(a0, X2, q)) % q
    alpha = np.random.randint(0, q, size=z0.shape, dtype=np.int64).astype(object)
    beta = np.random.randint(0, q, size=z0.shape, dtype=np.int64).astype(object)
    gamma = (-alpha - beta) % q
    reshared_feat = np.stack([(z0 + alpha) % q, (z1 + beta) % q, (z2 + gamma) % q], axis=-1)
    return reshared_feat.astype(np.int64)

def dot_product_matrix_rss_batch_gpu(a_shares_batch, X_shares_tensor, device, q=Q):
    a_t = torch.tensor(a_shares_batch, dtype=torch.int64, device=device) if not isinstance(a_shares_batch, torch.Tensor) else a_shares_batch
    a0, a1, a2 = a_t[..., 0], a_t[..., 1], a_t[..., 2]
    X0, X1, X2 = X_shares_tensor[..., 0], X_shares_tensor[..., 1], X_shares_tensor[..., 2]

    def compute_S_components_gpu(A_t, B_t):
        shift = 16
        mask = (1 << shift) - 1
        A0 = (A_t & mask).to(torch.float64)
        A1 = ((A_t >> shift) & mask).to(torch.float64)
        A2 = ((A_t >> (2 * shift)) & mask).to(torch.float64)
        A3 = (A_t >> (3 * shift)).to(torch.float64)
        B0 = (B_t & mask).to(torch.float64)
        B1 = ((B_t >> shift) & mask).to(torch.float64)
        B2 = ((B_t >> (2 * shift)) & mask).to(torch.float64)
        B3 = (B_t >> (3 * shift)).to(torch.float64)
        D00 = torch.matmul(A0, B0).round().to(torch.int64)
        D01 = torch.matmul(A0, B1).round().to(torch.int64)
        D02 = torch.matmul(A0, B2).round().to(torch.int64)
        D03 = torch.matmul(A0, B3).round().to(torch.int64)
        D10 = torch.matmul(A1, B0).round().to(torch.int64)
        D11 = torch.matmul(A1, B1).round().to(torch.int64)
        D12 = torch.matmul(A1, B2).round().to(torch.int64)
        D13 = torch.matmul(A1, B3).round().to(torch.int64)
        D20 = torch.matmul(A2, B0).round().to(torch.int64)
        D21 = torch.matmul(A2, B1).round().to(torch.int64)
        D22 = torch.matmul(A2, B2).round().to(torch.int64)
        D23 = torch.matmul(A2, B3).round().to(torch.int64)
        D30 = torch.matmul(A3, B0).round().to(torch.int64)
        D31 = torch.matmul(A3, B1).round().to(torch.int64)
        D32 = torch.matmul(A3, B2).round().to(torch.int64)
        D33 = torch.matmul(A3, B3).round().to(torch.int64)
        S0 = D00
        S1 = D01 + D10
        S2 = D02 + D11 + D20
        S3 = D03 + D12 + D21 + D30
        S4 = D13 + D22 + D31
        S5 = D23 + D32
        S6 = D33
        return S0, S1, S2, S3, S4, S5, S6

    def sum_S_tuples(*S_tuples):
        return tuple(sum(t) for t in zip(*S_tuples))

    S_z0 = sum_S_tuples(compute_S_components_gpu(a0, X0), compute_S_components_gpu(a0, X1), compute_S_components_gpu(a1, X0))
    S_z1 = sum_S_tuples(compute_S_components_gpu(a1, X1), compute_S_components_gpu(a1, X2), compute_S_components_gpu(a2, X1))
    S_z2 = sum_S_tuples(compute_S_components_gpu(a2, X2), compute_S_components_gpu(a2, X0), compute_S_components_gpu(a0, X2))

    def final_mod_Q_cpu(S_tuple):
        S = [s.cpu().numpy() for s in S_tuple]
        shift = 16
        C1 = (1 << shift) % q
        C2 = (1 << (2 * shift)) % q
        C3 = (1 << (3 * shift)) % q
        C4 = (1 << (4 * shift)) % q
        C5 = (1 << (5 * shift)) % q
        C6 = (1 << (6 * shift)) % q
        Z = (S[0].astype(object) +
             S[1].astype(object) * C1 +
             S[2].astype(object) * C2 +
             S[3].astype(object) * C3 +
             S[4].astype(object) * C4 +
             S[5].astype(object) * C5 +
             S[6].astype(object) * C6) % q
        return Z.astype(np.int64)

    z0_cpu = final_mod_Q_cpu(S_z0)
    z1_cpu = final_mod_Q_cpu(S_z1)
    z2_cpu = final_mod_Q_cpu(S_z2)
    alpha = np.random.randint(0, q, size=z0_cpu.shape, dtype=np.int64).astype(object)
    beta = np.random.randint(0, q, size=z0_cpu.shape, dtype=np.int64).astype(object)
    gamma = (-alpha - beta) % q
    reshared_feat = np.stack([(z0_cpu + alpha) % q, (z1_cpu + beta) % q, (z2_cpu + gamma) % q], axis=-1)
    return reshared_feat.astype(np.int64)

dot_product_matrix_rss_batch_GPU = dot_product_matrix_rss_batch_gpu

