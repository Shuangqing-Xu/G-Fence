import random
import numpy as np

from .constants import BASE, PRECISION_FRACTIONAL, PRECISION, Q, INVERSE

KAPPA = 6
INV_2 = pow(2, Q - 2, Q)
NEG_1 = Q - 1

assert (INVERSE * BASE**PRECISION_FRACTIONAL) % Q == 1
assert Q > BASE**(2 * PRECISION + KAPPA)

def encode(rational, precision_fractional=PRECISION_FRACTIONAL):
    if isinstance(rational, np.ndarray):
        return (rational * BASE**precision_fractional).astype(np.int64) % Q
    return int(rational * BASE**precision_fractional) % Q

def encode_list(rational, precision_fractional=PRECISION_FRACTIONAL):
    encoded = encode(rational, precision_fractional)
    if isinstance(encoded, np.ndarray):
        return encoded.tolist()
    return int(encoded)

def decode(field_element, precision_fractional=PRECISION_FRACTIONAL):
    upscaled = field_element if field_element <= Q / 2 else field_element - Q
    return upscaled / BASE**precision_fractional

def share(secret):
    first = random.randrange(Q)
    second = random.randrange(Q)
    third = (secret - first - second) % Q
    return [first, second, third]

def share_vector(secret_array):
    secret_array = np.asarray(secret_array, dtype=np.int64)
    first = np.random.randint(0, Q, secret_array.shape, dtype=np.int64)
    second = np.random.randint(0, Q, secret_array.shape, dtype=np.int64)
    third = (secret_array - first - second) % Q
    return np.stack([first, second, third], axis=-1)

def share_vector_positive(secret_array):
    secret_array = np.asarray(secret_array, dtype=np.int64)
    first = np.random.randint(0, Q, secret_array.shape, dtype=np.int64)
    second = np.random.randint(0, Q, secret_array.shape, dtype=np.int64)
    third = (secret_array + (2 * Q) - first - second) % Q
    return np.stack([first, second, third], axis=-1)

def share_vector_axis1(secret_array):
    secret_array = np.asarray(secret_array, dtype=np.int64)
    first = np.random.randint(0, Q, len(secret_array), dtype=np.int64)
    second = np.random.randint(0, Q, len(secret_array), dtype=np.int64)
    third = (secret_array - first - second) % Q
    return np.stack([first, second, third], axis=1)

def reconstruct(sharing):
    return sum(sharing) % Q

def reconstruct_array(sharing):
    sharing = np.array(sharing, dtype=object)
    return np.sum(sharing, axis=-1) % Q

def reconstruct_batch(sharing):
    return np.sum(sharing, axis=-1) % Q

def reshare(x):
    shares = [share(x[0]), share(x[1]), share(x[2])]
    return [sum(row) % Q for row in zip(*shares)]

def add(x, y):
    return (np.array(x, dtype=object) + np.array(y, dtype=object)) % Q

def sub(x, y):
    return (np.array(x, dtype=object) - np.array(y, dtype=object)) % Q

def imul(x, k):
    return [(xi * k) % Q for xi in x]

def imul_np(x_shares, scalar):
    return (np.array(x_shares, dtype=object) * (int(scalar) % Q)) % Q

def boolean_share_vector_batch(secret_array):
    secret_array = np.asarray(secret_array, dtype=np.int64)
    first = np.random.randint(0, 2, secret_array.shape, dtype=np.int64)
    second = np.random.randint(0, 2, secret_array.shape, dtype=np.int64)
    third = secret_array ^ first ^ second
    return np.stack([first, second, third], axis=-1)

def boolean_reconstruct_batch(sharing):
    return sharing[..., 0] ^ sharing[..., 1] ^ sharing[..., 2]

def apply_permutation_batch(x, pi):
    batch_indices = np.arange(pi.shape[0])[:, None]
    sorted_indices = np.argsort(pi, axis=1)
    return x[batch_indices, sorted_indices]

def inverse_permutation_batch(pi):
    return np.argsort(pi, axis=1)

def compose_perm_batch(p1, p2):
    batch_indices = np.arange(p1.shape[0])[:, None]
    return p1[batch_indices, p2]

def semi_shuffle_batch(pi_list, a, random_masks, is_boolean=False):
    shares = np.array(a)
    reshuffled_shares = np.empty_like(shares)

    def apply_shuffle_and_reshare(local_shares, perm, random_mask):
        for col in range(3):
            reshuffled_shares[:, :, col] = apply_permutation_batch(local_shares[:, :, col], perm)
        if is_boolean:
            return reshuffled_shares ^ random_mask
        return (reshuffled_shares + random_mask) % Q

    for idx, perm in enumerate(pi_list):
        shares = apply_shuffle_and_reshare(shares, perm, random_masks[idx])
    return shares

def B2A_batch(b_bool):
    batch_size, vector_size, _ = b_bool.shape
    r = np.random.randint(0, 2, (batch_size, vector_size), dtype=np.int64)
    r_bool = boolean_share_vector_batch(r)
    r_arith = share_vector(r)
    z_bool = b_bool ^ r_bool
    z = boolean_reconstruct_batch(z_bool)
    b_arith = np.zeros_like(r_arith, dtype=np.int64)
    for i in range(3):
        term = (2 * z * r_arith[:, :, i]) % Q
        if i == 0:
            b_arith[:, :, i] = (r_arith[:, :, i] + z - term) % Q
        else:
            b_arith[:, :, i] = (r_arith[:, :, i] - term) % Q
    return b_arith

def truncate_secure_ml_np(a, inverse=INVERSE):
    d = np.zeros_like(a, dtype=int)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tmp = a[i, j]
            sign = 1
            if tmp > Q // 2:
                sign = -1
                tmp = -tmp
            d[i, j] = (tmp * inverse) % Q
            d[i, j] = (d[i, j] * sign) % Q
    return d

def mul_np(x, y):
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)
    z0 = (x[:, 0] * y[:, 0] + x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]) % Q
    z1 = (x[:, 1] * y[:, 1] + x[:, 1] * y[:, 2] + x[:, 2] * y[:, 1]) % Q
    z2 = (x[:, 2] * y[:, 2] + x[:, 2] * y[:, 0] + x[:, 0] * y[:, 2]) % Q
    z = np.array([z0, z1, z2]).T
    random_masks = np.random.randint(0, Q, (len(x), 3), dtype=np.int64)
    random_masks[:, 2] = (-random_masks[:, 0] - random_masks[:, 1]) % Q
    return truncate_secure_ml_np((z + random_masks) % Q)

def mul_exact_np(x, y):
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)
    z0 = (x[:, 0] * y[:, 0] + x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]) % Q
    z1 = (x[:, 1] * y[:, 1] + x[:, 1] * y[:, 2] + x[:, 2] * y[:, 1]) % Q
    z2 = (x[:, 2] * y[:, 2] + x[:, 2] * y[:, 0] + x[:, 0] * y[:, 2]) % Q
    z = np.array([z0, z1, z2]).T
    random_masks = np.random.randint(0, Q, (len(x), 3), dtype=np.int64)
    random_masks[:, 2] = (-random_masks[:, 0] - random_masks[:, 1]) % Q
    return (z + random_masks) % Q

def mod_sum_manual(arr):
    arr = np.array(arr)
    column_sums = [0] * arr.shape[1]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            column_sums[j] += int(arr[i, j])
    return [value % Q for value in column_sums]

class SecureRationalVector:
    def __init__(self):
        self.shares = None

    @staticmethod
    def secure(secret):
        z = SecureRationalVector()
        if isinstance(secret, (int, float)):
            z.shares = share(encode(secret))
        else:
            z.shares = share_vector_axis1(encode(secret))
        return z

    def reveal(self):
        decoded_values = [decode(reconstruct(sharing)) for sharing in self.shares]
        return np.array(decoded_values)

    def __repr__(self):
        return f"SecureRationalVector({self.reveal()})"

    def __add__(self, other):
        assert self.shares.shape == other.shares.shape, "Shapes must match for addition"
        z = SecureRationalVector()
        z.shares = add(self.shares, other.shares)
        return z

    def __sub__(self, other):
        z = SecureRationalVector()
        z.shares = sub(self.shares, other.shares)
        return z

    def __mul__(self, other):
        z = SecureRationalVector()
        z.shares = mul_np(self.shares, other.shares)
        return z

    def __pow__(self, exponent):
        z = SecureRationalVector.secure(np.ones(self.shares.shape[0]))
        for _ in range(exponent):
            z = z * self
        return z

