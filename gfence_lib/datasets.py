import numpy as np
import torch
from torch_geometric.datasets import Planetoid, KarateClub, WebKB

def load_data(dataset_name, root="../data"):
    if dataset_name == "KarateClub":
        dataset = KarateClub()
        return dataset, dataset[0]
    if dataset_name in ["Cora", "Citeseer", "PubMed"]:
        dataset = Planetoid(root=root, name=dataset_name)
        return dataset, dataset[0]
    if dataset_name in ["Texas", "Cornell", "Wisconsin"]:
        dataset = WebKB(root=root, name=dataset_name)
        return dataset, dataset[0]
    raise ValueError(f"Dataset {dataset_name} not supported.")

def row_normalize(x, eps=1e-12):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / (norm + eps)

def random_node_split(num_nodes, train_ratio, val_ratio, test_ratio, seed=42):
    torch.manual_seed(seed)
    perm = torch.randperm(num_nodes)
    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask

def partition_data(data, num_clients):
    indices = np.random.permutation(data.num_nodes)
    split_indices = np.array_split(indices, num_clients)
    return [torch.tensor(idx, dtype=torch.long) for idx in split_indices]

