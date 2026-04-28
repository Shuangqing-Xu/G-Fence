import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gfence_lib.dice import DiceEnsembleSampler, generate_dg_props

if __name__ == "__main__":
    SIGMA = 1000.0
    N_LAYERS = 64
    TABLE_SIZE = 2**16
    N_SAMPLES = 1000000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    target_pmf, support_vals = generate_dg_props(SIGMA, r=int(SIGMA * 20), L=int(SIGMA * 10), device=DEVICE)
    if TABLE_SIZE < 2 * len(target_pmf):
        TABLE_SIZE = 2 * len(target_pmf)
    sampler = DiceEnsembleSampler(target_pmf, support_vals, TABLE_SIZE, device=DEVICE)
    sampler.build_tables(N_LAYERS)
    noise = sampler.sample_noise_one_hot_logic(N_SAMPLES)
    print(f"Sample Mean: {noise.float().mean().item():.4f}")
    print(f"Sample Std: {noise.float().std().item():.4f}")

