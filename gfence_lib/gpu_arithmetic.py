import torch
import numpy as np

from .constants import BASE, PRECISION_FRACTIONAL, Q, INVERSE

def encode_gpu(rational_tensor, device, precision_fractional=PRECISION_FRACTIONAL):
    upscaled = (rational_tensor * (BASE**precision_fractional)).round().to(torch.int64)
    return upscaled % Q

def sum_mod_Q_gpu(tensor, dim, q=Q):
    lo = tensor & 0x7FFFFFFF
    hi = tensor >> 31
    sum_lo = lo.sum(dim=dim, dtype=torch.int64)
    sum_hi = hi.sum(dim=dim, dtype=torch.int64)
    sum_hi_hi = sum_hi >> 30
    sum_hi_lo = sum_hi & 0x3FFFFFFF
    return (sum_hi_hi + (sum_hi_lo << 31) + sum_lo) % q

def radix_split_mult_gpu(A_t, B_t, op="matmul", q=Q):
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
    mul_fn = (lambda a, b: torch.matmul(a, b).round().to(torch.int64)) if op == "matmul" else (lambda a, b: (a * b).round().to(torch.int64))
    D00 = mul_fn(A0, B0)
    D01 = mul_fn(A0, B1)
    D02 = mul_fn(A0, B2)
    D03 = mul_fn(A0, B3)
    D10 = mul_fn(A1, B0)
    D11 = mul_fn(A1, B1)
    D12 = mul_fn(A1, B2)
    D13 = mul_fn(A1, B3)
    D20 = mul_fn(A2, B0)
    D21 = mul_fn(A2, B1)
    D22 = mul_fn(A2, B2)
    D23 = mul_fn(A2, B3)
    D30 = mul_fn(A3, B0)
    D31 = mul_fn(A3, B1)
    D32 = mul_fn(A3, B2)
    D33 = mul_fn(A3, B3)
    S0 = D00
    S1 = D01 + D10
    S2 = D02 + D11 + D20
    S3 = D03 + D12 + D21 + D30
    S4 = D13 + D22 + D31
    S5 = D23 + D32
    S6 = D33
    C1 = (1 << shift) % q
    C2 = (1 << (2 * shift)) % q
    C3 = (1 << (3 * shift)) % q
    C4 = (1 << (4 * shift)) % q
    C5 = (1 << (5 * shift)) % q
    C6 = (1 << (6 * shift)) % q
    S = [s.cpu().numpy() for s in (S0, S1, S2, S3, S4, S5, S6)]
    Z = (S[0].astype(object) +
         S[1].astype(object) * C1 +
         S[2].astype(object) * C2 +
         S[3].astype(object) * C3 +
         S[4].astype(object) * C4 +
         S[5].astype(object) * C5 +
         S[6].astype(object) * C6) % q
    return torch.tensor(Z.astype(np.int64), device=A_t.device)

def truncate_secure_ml_gpu(a, q, device, inverse=INVERSE):
    inverse_t = torch.tensor(inverse, dtype=torch.int64, device=device)
    is_negative = a > (q // 2)
    tmp = torch.where(is_negative, -a, a)
    d = radix_split_mult_gpu(tmp, inverse_t, op="mul", q=q)
    return torch.where(is_negative, -d, d) % q

def mul_gpu(X_t, Y_t, q, device, inverse=INVERSE):
    x0, x1, x2 = X_t[..., 0], X_t[..., 1], X_t[..., 2]
    y0, y1, y2 = Y_t[..., 0], Y_t[..., 1], Y_t[..., 2]

    def rad_mul(A, B):
        return radix_split_mult_gpu(A, B, op="mul", q=q)

    z0 = (rad_mul(x0, y0) + rad_mul(x0, y1) + rad_mul(x1, y0)) % q
    z1 = (rad_mul(x1, y1) + rad_mul(x1, y2) + rad_mul(x2, y1)) % q
    z2 = (rad_mul(x2, y2) + rad_mul(x2, y0) + rad_mul(x0, y2)) % q
    Z = torch.stack([z0, z1, z2], dim=-1)
    m0 = torch.randint(0, q, Z.shape[:-1], dtype=torch.int64, device=device)
    m1 = torch.randint(0, q, Z.shape[:-1], dtype=torch.int64, device=device)
    m2 = ((2 * q) - m0 - m1) % q
    reshared = (Z + torch.stack([m0, m1, m2], dim=-1)) % q
    return truncate_secure_ml_gpu(reshared, q, device, inverse=inverse)

def fast_mul_mod_Q_gpu(A, B, q=Q):
    A0 = A & 0x7FFFFFFF
    A1 = A >> 31
    B0 = B & 0x7FFFFFFF
    B1 = B >> 31
    P0 = A0 * B0
    P1 = A0 * B1 + A1 * B0
    P2 = A1 * B1
    P1_lo = P1 & 0x3FFFFFFF
    P1_hi = P1 >> 30
    res = P0 % q
    res = (res + (P1_lo << 31)) % q
    res = (res + P1_hi) % q
    res = (res + (P2 << 1)) % q
    return res

def mul_gpu_int(X_t, Y_t, q, device):
    x0, x1, x2 = X_t[..., 0], X_t[..., 1], X_t[..., 2]
    y0, y1, y2 = Y_t[..., 0], Y_t[..., 1], Y_t[..., 2]
    z0 = (fast_mul_mod_Q_gpu(x0, y0, q) + fast_mul_mod_Q_gpu(x0, y1, q) + fast_mul_mod_Q_gpu(x1, y0, q)) % q
    z1 = (fast_mul_mod_Q_gpu(x1, y1, q) + fast_mul_mod_Q_gpu(x1, y2, q) + fast_mul_mod_Q_gpu(x2, y1, q)) % q
    z2 = (fast_mul_mod_Q_gpu(x2, y2, q) + fast_mul_mod_Q_gpu(x2, y0, q) + fast_mul_mod_Q_gpu(x0, y2, q)) % q
    Z = torch.stack([z0, z1, z2], dim=-1)
    m0 = torch.randint(0, q, Z.shape[:-1], dtype=torch.int64, device=device)
    m1 = torch.randint(0, q, Z.shape[:-1], dtype=torch.int64, device=device)
    m2 = ((2 * q) - m0 - m1) % q
    return (Z + torch.stack([m0, m1, m2], dim=-1)) % q

def share_tensor_gpu(plain_tensor, q, device):
    first = torch.randint(0, q, plain_tensor.shape, dtype=torch.int64, device=device)
    second = torch.randint(0, q, plain_tensor.shape, dtype=torch.int64, device=device)
    third = (plain_tensor + (2 * q) - first - second) % q
    return torch.stack([first, second, third], dim=-1)

