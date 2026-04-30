<div align="center">

<h1>G-Fence<br/>Secure, Efficient and Differentially Private Graph Neural Network Training over Federated Graphs</h1>

</div>

This repository contains a prototype implementation of the protocols proposed in:

> **G-Fence: Secure, Efficient and Differentially Private Graph Neural Network Training over Federated Graphs**

G-Fence is a framework for privacy-preserving GNN training over **federated graphs**, where graph data are distributed across multiple clients and cross-client edges are needed for model utility. The system combines a decoupled two-phase GNN training paradigm with replicated secret sharing, sparsity-aware topology sharing, secure discrete Gaussian noise sampling, and lightweight integrity checks. It supports both **semi-honest** and **malicious-secure** server-side adversary models while providing **edge-level differential privacy** for the released embeddings and trained model.

---

## Table of Contents

- [0 Backgrounds](#0-backgrounds)
- [1 Code Organization](#1-code-organization)
- [2 Quick Set Up](#2-quick-set-up)
- [3 Running the Artifact](#3-running-the-artifact)
- [4 Configuration](#4-configuration)

---

## 0 Backgrounds

### 0.1 Two-Phase GNN Training

G-Fence decouples GNN training into:

1. **Secure pre-training**: the servers compute graph-propagated node embeddings from secret-shared topology and features.
2. **Iterative model training**: clients train a lightweight classifier on the pre-computed embeddings using FedAvg-style aggregation.

This avoids repeatedly sharing node embeddings during iterative GNN training.

### 0.2 Threat Model and Privacy Goal

The framework uses three non-colluding servers from different trust domains. The default security model is honest majority: each server may try to infer private information, and at most one server may maliciously deviate from the protocol.

G-Fence aims to:

- protect each client's local subgraph and features from the servers,
- use cross-client graph information without revealing raw topology,
- provide edge-level differential privacy for released embeddings and the trained model,
- detect malicious server-side deviations and abort when integrity checks fail.

---

## 1 Code Organization

```text
.
├── DP_SecNoiseSamp/
│   ├── dice_ensemble_gpu.py
│   │   Entry point for Dice Ensemble / LUT generation.
│   ├── noise_sampling_gpu_semi_accelerated.py
│   │   Entry point for semi-honest secure noise sampling.
│   ├── noise_sampling_gpu_mali_accelerated.py
│   │   Entry point for malicious-secure secure noise sampling.
│   └── G-Fence-DP.py
│       Entry point for secure GNN training with DP noise injection.
├── G-Fence/
│   ├── G-Fence-Semi.py
│   │   Entry point for the semi-honest G-Fence workflow.
│   ├── G-Fence-Mali.py
│   │   Entry point for the malicious-secure G-Fence workflow.
│   └── rss_mult_verify.py
│       Compatibility wrapper for RSS verification utilities.
├── gfence_lib/
│   ├── constants.py
│   │   Field and fixed-point parameters.
│   ├── rss.py
│   │   RSS sharing, reconstruction, shuffle, B2A, and fixed-point helpers.
│   ├── crypto.py
│   │   AES-CTR PRG and MAC helper routines.
│   ├── dot_products.py
│   │   CPU/GPU RSS matrix-dot-product kernels.
│   ├── gpu_arithmetic.py
│   │   GPU modular arithmetic, radix splitting, and RSS multiplication kernels.
│   ├── dice.py
│   │   Dice Ensemble table construction and plaintext LUT sampling.
│   ├── noise_sampling.py
│   │   Semi-honest and malicious-secure secure noise sampling protocols.
│   ├── verification.py
│   │   Dot-product verification used by the malicious-secure workflow.
│   ├── datasets.py
│   │   Dataset loading, splitting, normalization, and client partitioning.
│   ├── federated.py
│   │   Client, model, optimizer, and evaluation helpers.
│   ├── pretrainers.py
│   │   Semi-honest, malicious-secure, and DP-injected secure pre-training.
│   └── training.py
│       Shared federated training loop.
├── data/
│   ├── Cora/
│   ├── Citeseer/
│   └── PubMed/
├── requirements.txt
└── README.md
```

---

## 2 Quick Set Up

### 2.1 Environmental Requirements

**Hardware**

- CPU execution is supported for small sanity runs.
- A CUDA-capable GPU is recommended for the GPU-accelerated secure aggregation and noise-sampling paths.
- Larger datasets such as PubMed require substantially more memory than Cora/Citeseer.

**Software**

- Python 3.9 or newer.
- PyTorch.
- PyTorch Geometric.
- PyCryptodome.
- NumPy.

### 2.2 Installation

Create an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch and PyTorch Geometric wheels that match your CUDA version if GPU acceleration is used.

---

## 3 Running the Artifact

Run each script from its own directory, because the prototype uses relative paths for local imports and datasets.

### 3.1 Semi-Honest G-Fence

```bash
cd G-Fence
python G-Fence-Semi.py
```

Expected output format:

```text
Round 001 | Val Acc (Server): ... | Test Acc (Server): ... | Test Acc (Client Avg): ...
Round 010 | Val Acc (Server): ... | Test Acc (Server): ... | Test Acc (Client Avg): ...
```

### 3.2 Malicious-Secure G-Fence

```bash
cd G-Fence
python G-Fence-Mali.py
```

This path enables server-side integrity checks for malicious-security evaluation. It reports the same accuracy format as the semi-honest workflow.

### 3.3 DP-Injected G-Fence

```bash
cd DP_SecNoiseSamp
python G-Fence-DP.py
```

This script integrates secure GNN training with secure DP noise injection.

### 3.4 LUT Generation and Secure Noise Sampling

```bash
cd DP_SecNoiseSamp
python dice_ensemble_gpu.py
python noise_sampling_gpu_semi_accelerated.py
python noise_sampling_gpu_mali_accelerated.py
```

Expected output format:

```text
Sample Mean: ...
Sample Std: ...
```

---

## 4 Configuration

Experiment parameters are defined near the top of each training script in the `ARGS` dictionary.

Common options:

- `dataset`: `KarateClub`, `Cora`, `Citeseer`, `PubMed`, `Texas`, `Cornell`, or `Wisconsin`.
- `num_clients`: number of federated clients.
- `hops`: graph propagation depth.
- `global_rounds`: number of FedAvg rounds.
- `local_epochs`: local training epochs per round.
- `batch_size`: secure aggregation batch size.
- `lr`: learning rate.
- `weight_decay`: optimizer weight decay.
- `split_seed`: random seed for train/validation/test split.
- `dp_sigma`: DP noise scale in `DP_SecNoiseSamp/G-Fence-DP.py`.

The paper evaluates G-Fence on Cora, Citeseer, and PubMed. The corresponding dataset folders are included under `data/`, and PyTorch Geometric loaders use `../data` as the root path when scripts are launched from `DP_SecNoiseSamp/` or `G-Fence/`.

Paper-scale dataset settings:

| Dataset | Nodes | Edges | Max Degree | Feature Dim. | Classes | Clients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 2,708 | 5,429 | 169 | 1,433 | 7 | 10 |
| Citeseer | 3,327 | 4,732 | 100 | 3,703 | 6 | 20 |
| PubMed | 19,717 | 44,338 | 171 | 500 | 3 | 300 |
