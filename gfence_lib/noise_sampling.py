import random
import numpy as np
import torch

from .constants import BASE, PRECISION_FRACTIONAL, Q, inverse_for_precision
from .gpu_arithmetic import encode_gpu, sum_mod_Q_gpu, radix_split_mult_gpu, mul_gpu, mul_gpu_int, share_tensor_gpu
from .rss import reconstruct_array, decode

def generate_onehot_shares_shuffle_gpu(n_samples, table_size, q, device):
    base_vectors = torch.zeros((n_samples, table_size), dtype=torch.int64, device=device)
    base_vectors[:, 0] = 1
    first = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
    second = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
    third = (base_vectors + (2 * q) - first - second) % q
    shares = torch.stack([first, second, third], dim=-1)
    for _ in range(3):
        pi = torch.rand((n_samples, table_size), device=device).argsort(dim=1)
        pi_expanded = pi.unsqueeze(-1).expand(-1, -1, 3)
        reshuffled = torch.gather(shares, 1, pi_expanded)
        m0 = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
        m1 = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
        m2 = ((2 * q) - m0 - m1) % q
        shares = (reshuffled + torch.stack([m0, m1, m2], dim=-1)) % q
    return shares

def generate_onehot_shares_shuffle_gpu_inplace(n_samples, table_size, q, device):
    shares = torch.empty((n_samples, table_size, 3), dtype=torch.int64, device=device)
    shares[..., 0].random_(0, q)
    shares[..., 1].random_(0, q)
    shares[..., 2] = 0
    shares[:, 0, 2] = 1
    shares[..., 2] = (shares[..., 2] + (2 * q) - shares[..., 0] - shares[..., 1]) % q
    for _ in range(3):
        pi = torch.rand((n_samples, table_size), device=device).argsort(dim=1)
        pi_expanded = pi.unsqueeze(-1).expand(-1, -1, 3)
        reshuffled = torch.gather(shares, 1, pi_expanded)
        del pi, pi_expanded
        shares[..., 0].random_(0, q)
        shares[..., 1].random_(0, q)
        shares[..., 2] = ((2 * q) - shares[..., 0] - shares[..., 1]) % q
        shares.add_(reshuffled).remainder_(q)
        del reshuffled
    return shares

def generate_onehot_shares_shuffle_gpu_malicious(n_samples, table_size, r_shares, q, device):
    r0, r1, r2 = r_shares
    v_s0 = torch.zeros((n_samples, table_size), dtype=torch.int64, device=device)
    v_s0[:, 0] = 1
    mac_s0 = torch.zeros((n_samples, table_size), dtype=torch.int64, device=device)
    mac_s0[:, 0] = r0
    v_s1 = torch.zeros((n_samples, table_size), dtype=torch.int64, device=device)
    mac_s1 = torch.zeros((n_samples, table_size), dtype=torch.int64, device=device)
    mac_s1[:, 0] = r1
    v_s2 = torch.zeros((n_samples, table_size), dtype=torch.int64, device=device)
    mac_s2 = torch.zeros((n_samples, table_size), dtype=torch.int64, device=device)
    mac_s2[:, 0] = r2
    shares = torch.stack([v_s0, v_s1, v_s2], dim=-1)
    mac_shares = torch.stack([mac_s0, mac_s1, mac_s2], dim=-1)
    for _ in range(3):
        pi = torch.rand((n_samples, table_size), device=device).argsort(dim=1)
        pi_expanded = pi.unsqueeze(-1).expand(-1, -1, 3)
        reshuffled_v = torch.gather(shares, 1, pi_expanded)
        reshuffled_mac = torch.gather(mac_shares, 1, pi_expanded)
        m0_v = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
        m1_v = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
        m2_v = ((2 * q) - m0_v - m1_v) % q
        m0_mac = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
        m1_mac = torch.randint(0, q, (n_samples, table_size), dtype=torch.int64, device=device)
        m2_mac = ((2 * q) - m0_mac - m1_mac) % q
        shares = (reshuffled_v + torch.stack([m0_v, m1_v, m2_v], dim=-1)) % q
        mac_shares = (reshuffled_mac + torch.stack([m0_mac, m1_mac, m2_mac], dim=-1)) % q
    return shares, mac_shares

def verify_macs_gpu(v_shares, mac_shares, r_shares, q, device):
    N, T, _ = v_shares.shape
    alphas_plain = torch.randint(0, q, (N, T), dtype=torch.int64, device=device)
    alphas_shares = share_tensor_gpu(alphas_plain, q, device)
    alpha_mul_v = mul_gpu_int(alphas_shares, v_shares, q, device)
    alpha_mul_mac = mul_gpu_int(alphas_shares, mac_shares, q, device)
    u_shares = sum_mod_Q_gpu(alpha_mul_v.view(-1, 3), dim=0, q=q)
    v_shares_sum = sum_mod_Q_gpu(alpha_mul_mac.view(-1, 3), dim=0, q=q)
    r_shares_t = torch.tensor(r_shares, dtype=torch.int64, device=device).unsqueeze(0)
    r_u_shares = mul_gpu_int(r_shares_t, u_shares.unsqueeze(0), q, device).squeeze(0)
    w_shares = (r_u_shares - v_shares_sum + q) % q
    w_rec = sum_mod_Q_gpu(w_shares.unsqueeze(0), dim=1, q=q).item()
    return w_rec == 0

def secure_noise_sampling_gpu(tables_np, support_vals_np, n_samples, device, q=Q, precision_fractional=PRECISION_FRACTIONAL, support_is_encoded=False, scale_factor=None, inverse=None, inplace_onehot=False):
    n_layers = tables_np.shape[0]
    table_size = tables_np.shape[1]
    tables = torch.tensor(tables_np, dtype=torch.int64, device=device)
    if support_is_encoded:
        support_vals = torch.tensor(support_vals_np, dtype=torch.int64, device=device)
        encoded_support = support_vals % q
    else:
        support_vals = torch.tensor(support_vals_np, dtype=torch.float64, device=device)
        encoded_support = encode_gpu(support_vals, device, precision_fractional=precision_fractional)
    inverse = inverse if inverse is not None else inverse_for_precision(precision_fractional)
    scale_t = torch.tensor(scale_factor if scale_factor is not None else BASE**precision_fractional, dtype=torch.int64, device=device)
    y_next = torch.zeros((n_samples, 3), dtype=torch.int64, device=device)
    onehot_fn = generate_onehot_shares_shuffle_gpu_inplace if inplace_onehot else generate_onehot_shares_shuffle_gpu
    for i in reversed(range(n_layers)):
        table = tables[i]
        x_i = (table != -1).sum().item()
        onehot_shares = onehot_fn(n_samples, table_size, q, device)
        if x_i < table_size:
            c_p = sum_mod_Q_gpu(onehot_shares[:, x_i:, :], dim=1, q=q)
        else:
            c_p = torch.zeros((n_samples, 3), dtype=torch.int64, device=device)
        c_p_fixed = radix_split_mult_gpu(c_p, scale_t, op="mul", q=q)
        term1 = torch.zeros((n_samples, 3), dtype=torch.int64, device=device)
        if x_i > 0:
            valid_entries = table[:x_i]
            encoded_table_values = encoded_support[valid_entries]
            valid_onehot = onehot_shares[:, :x_i, :]
            for s in range(3):
                term1[:, s] = radix_split_mult_gpu(valid_onehot[:, :, s], encoded_table_values, op="matmul", q=q)
        term2 = mul_gpu(c_p_fixed, y_next, q, device, inverse=inverse)
        y_next = (term1 + term2) % q
        if device != "cpu" and torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return y_next.cpu().numpy()

def secure_noise_sampling_gpu_malicious(tables_np, support_vals_np, n_samples, device, q=Q, precision_fractional=15):
    n_layers = tables_np.shape[0]
    table_size = tables_np.shape[1]
    tables = torch.tensor(tables_np, dtype=torch.int64, device=device)
    support_vals = torch.tensor(support_vals_np, dtype=torch.float64, device=device)
    encoded_support = encode_gpu(support_vals, device, precision_fractional=precision_fractional)
    inverse = inverse_for_precision(precision_fractional)
    scale_t = torch.tensor(BASE**precision_fractional, dtype=torch.int64, device=device)
    y_next = torch.zeros((n_samples, 3), dtype=torch.int64, device=device)
    global_r = random.randint(1, q - 1)
    r_s0 = random.randint(0, q - 1)
    r_s1 = random.randint(0, q - 1)
    r_shares = (r_s0, r_s1, (global_r - r_s0 - r_s1) % q)
    for i in reversed(range(n_layers)):
        table = tables[i]
        x_i = (table != -1).sum().item()
        onehot_shares, mac_shares = generate_onehot_shares_shuffle_gpu_malicious(n_samples, table_size, r_shares, q, device)
        assert verify_macs_gpu(onehot_shares, mac_shares, r_shares, q, device), "MAC verification failed."
        if x_i < table_size:
            c_p = sum_mod_Q_gpu(onehot_shares[:, x_i:, :], dim=1, q=q)
        else:
            c_p = torch.zeros((n_samples, 3), dtype=torch.int64, device=device)
        c_p_fixed = radix_split_mult_gpu(c_p, scale_t, op="mul", q=q)
        term1 = torch.zeros((n_samples, 3), dtype=torch.int64, device=device)
        if x_i > 0:
            valid_entries = table[:x_i]
            encoded_table_values = encoded_support[valid_entries]
            valid_onehot = onehot_shares[:, :x_i, :]
            for s in range(3):
                term1[:, s] = radix_split_mult_gpu(valid_onehot[:, :, s], encoded_table_values, op="matmul", q=q)
        term2 = mul_gpu(c_p_fixed, y_next, q, device, inverse=inverse)
        y_next = (term1 + term2) % q
    return y_next.cpu().numpy()

def get_dp_noise_shares(n_total, tables_np, support_vals_np, q, scale_factor, chunk_size=512):
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)] if num_gpus > 0 else [torch.device("cpu")]
    all_noise = []
    for i, start_idx in enumerate(range(0, n_total, chunk_size)):
        cur_device = devices[i % len(devices)]
        cur_n = min(chunk_size, n_total - start_idx)
        noise_chunk = secure_noise_sampling_gpu(
            tables_np,
            support_vals_np,
            cur_n,
            cur_device,
            q=q,
            support_is_encoded=True,
            scale_factor=scale_factor,
            inplace_onehot=True,
        )
        all_noise.append(noise_chunk)
    return np.concatenate(all_noise, axis=0)

def plaintext_noise_sampling(tables, support_vals, n_samples):
    n_layers = tables.shape[0]
    table_size = tables.shape[1]
    y_next = np.zeros(n_samples)
    for i in reversed(range(n_layers)):
        table = tables[i]
        valid_indices = table[table != -1]
        x_i = len(valid_indices)
        indices = np.random.randint(0, table_size, size=n_samples)
        onehot = np.zeros((n_samples, table_size))
        onehot[np.arange(n_samples), indices] = 1
        c_p = onehot[:, x_i:].sum(axis=1) if x_i < table_size else np.zeros(n_samples)
        if x_i > 0:
            term1 = np.zeros(n_samples)
            table_values = support_vals[valid_indices]
            for k in range(x_i):
                term1 += onehot[:, k] * table_values[k]
        else:
            term1 = np.zeros(n_samples)
        y_next = term1 + c_p * y_next
    return y_next

def reconstruct_noise(noise_shares, precision_fractional=PRECISION_FRACTIONAL):
    noise_rec = reconstruct_array(noise_shares)
    return np.array([decode(x, precision_fractional=precision_fractional) for x in noise_rec])

