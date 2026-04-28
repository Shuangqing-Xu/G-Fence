import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gfence_lib.dice import DiceEnsembleSampler, generate_dg_props
from gfence_lib.noise_sampling import secure_noise_sampling_gpu_malicious, reconstruct_noise

if __name__ == "__main__":
    SIGMA = 2**10
    N_SAMPLES = 1000
    N_LAYERS = 40
    TABLE_SIZE = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    target_pmf, support_vals = generate_dg_props(SIGMA, r=int(SIGMA * 20), L=int(SIGMA * 10), device=DEVICE)
    if TABLE_SIZE < 2 * len(target_pmf):
        TABLE_SIZE = 2 * len(target_pmf)
    sampler = DiceEnsembleSampler(target_pmf, support_vals, TABLE_SIZE, device=DEVICE)
    sampler.build_tables(N_LAYERS)
    tables_np = sampler.tables.cpu().numpy()
    support_vals_np = support_vals.cpu().numpy()
    noise_shares = secure_noise_sampling_gpu_malicious(tables_np, support_vals_np, N_SAMPLES, DEVICE)
    sampled_noise = reconstruct_noise(noise_shares, precision_fractional=15)
    print(f"Sample Std: {np.std(sampled_noise):.2f}")
    print(f"Sample Mean: {np.mean(sampled_noise):.2f}")