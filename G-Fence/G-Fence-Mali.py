import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gfence_lib.pretrainers import MaliciousPreTrainer
from gfence_lib.training import run_federated_sgc

ARGS = {
    "dataset": "Cora",
    "num_clients": 50,
    "hops": 2,
    "global_rounds": 100,
    "local_epochs": 15,
    "batch_size": 256,
    "lr": 0.01,
    "weight_decay": 1e-5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "split_seed": 44,
    "dp_sigma": 0,
    "eps": 1e-12,
}

if __name__ == "__main__":
    run_federated_sgc(ARGS, MaliciousPreTrainer)

