import numpy as np
from Crypto.Cipher import AES

from .constants import Q

def aes_ctr_prg(key, counter, n):
    cipher = AES.new(key, AES.MODE_CTR, nonce=b"", initial_value=counter.to_bytes(16, byteorder="big"))
    random_bytes = cipher.encrypt(b"\x00" * (16 * n))
    random_numbers = np.frombuffer(random_bytes, dtype=np.uint64)[:n]
    return [int(num) % Q for num in random_numbers]

def compute_mac(encoded_x, mac_key):
    assert len(encoded_x) == len(mac_key), "MAC inputs must have the same length"
    return [(key * value) % Q for key, value in zip(mac_key, encoded_x)]

