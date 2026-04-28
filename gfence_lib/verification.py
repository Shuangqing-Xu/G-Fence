import random
import numpy as np

from .constants import Q
from .rss import add, sub, share, reconstruct, imul_np, INV_2

def f_coin():
    return random.randrange(1, Q)

def dot_product_rss_optimized(x, y):
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)
    z0 = (x[:, 0] * y[:, 0] + x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]) % Q
    z1 = (x[:, 1] * y[:, 1] + x[:, 1] * y[:, 2] + x[:, 2] * y[:, 1]) % Q
    z2 = (x[:, 2] * y[:, 2] + x[:, 2] * y[:, 0] + x[:, 0] * y[:, 2]) % Q
    S0 = np.sum(z0) % Q
    S1 = np.sum(z1) % Q
    S2 = np.sum(z2) % Q
    alpha = random.randrange(Q)
    beta = random.randrange(Q)
    gamma = (-alpha - beta) % Q
    return np.array([(S0 + alpha) % Q, (S1 + beta) % Q, (S2 + gamma) % Q]).reshape(1, 3)

class Protocol2Verifier:
    def check_triple(self, x, y, z):
        prod = dot_product_rss_optimized(x, y)
        diff = sub(prod[0], z[0])
        return reconstruct(diff) == 0

    def check_triple_MAESTRO(self, x, y, z):
        x_prime = np.array(share(random.randrange(Q))).reshape(1, 3)
        z_prime = dot_product_rss_optimized(x_prime, y)
        t = f_coin()
        rho_shares = add(x, imul_np(x_prime, t))
        rho = reconstruct(rho_shares[0])
        left = add(z, imul_np(z_prime, t))
        right = imul_np(y, rho)
        sigma = sub(left, right)
        return reconstruct(sigma[0]) == 0

    def verify_dot_product(self, X_shares, Y_shares, Z_shares):
        N = X_shares.shape[0]
        if N == 1:
            return self.check_triple_MAESTRO(X_shares, Y_shares, Z_shares)
        assert N % 2 == 0, "Input length must be power of 2"
        X_even = X_shares[0::2]
        X_odd = X_shares[1::2]
        Y_even = Y_shares[0::2]
        Y_odd = Y_shares[1::2]
        X_2 = sub(imul_np(X_odd, 2), X_even)
        Y_2 = sub(imul_np(Y_odd, 2), Y_even)
        H1_shares = dot_product_rss_optimized(X_odd, Y_odd)
        H2_shares = dot_product_rss_optimized(X_2, Y_2)
        H0_shares = sub(Z_shares, H1_shares)
        r = f_coin()
        r_inv_coef = (1 - r) % Q
        X_next = add(imul_np(X_even, r_inv_coef), imul_np(X_odd, r))
        Y_next = add(imul_np(Y_even, r_inv_coef), imul_np(Y_odd, r))
        l0 = ((r - 1) * (r - 2) * INV_2) % Q
        l1 = (-r * (r - 2)) % Q
        l2 = (r * (r - 1) * INV_2) % Q
        Z_next = add(add(imul_np(H0_shares, l0), imul_np(H1_shares, l1)), imul_np(H2_shares, l2))
        return self.verify_dot_product(X_next, Y_next, Z_next)

    def verify_batch(self, x_vectors, y_vectors, z_scalars):
        X_batch_list = []
        Y_batch_list = []
        Z_batch_acc = np.zeros((1, 3), dtype=int)
        r = f_coin()
        r_pow = 1
        for i in range(len(x_vectors)):
            x_part = imul_np(x_vectors[i], r_pow)
            X_batch_list.append(x_part)
            Y_batch_list.append(y_vectors[i])
            z_part = imul_np(z_scalars[i].reshape(1, 3), r_pow)
            Z_batch_acc = add(Z_batch_acc, z_part)
            r_pow = (r_pow * r) % Q
        X_big = np.concatenate(X_batch_list, axis=0)
        Y_big = np.concatenate(Y_batch_list, axis=0)
        total_len = X_big.shape[0]
        next_pow2 = 1
        while next_pow2 < total_len:
            next_pow2 *= 2
        if next_pow2 > total_len:
            padding = next_pow2 - total_len
            zeros = np.zeros((padding, 3), dtype=int)
            X_big = np.concatenate([X_big, zeros], axis=0)
            Y_big = np.concatenate([Y_big, zeros], axis=0)
        return self.verify_dot_product(X_big, Y_big, Z_batch_acc)

