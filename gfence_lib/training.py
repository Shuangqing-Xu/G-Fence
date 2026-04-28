import copy
import torch

from .datasets import load_data, random_node_split, partition_data
from .federated import Client, SimpleLinearModel, server_evaluate

def run_federated_sgc(args, pretrainer_cls):
    dataset, data = load_data(args["dataset"])
    data = data.to(args["device"])
    num_classes = dataset.num_classes
    train_mask, val_mask, test_mask = random_node_split(
        num_nodes=data.num_nodes,
        train_ratio=args["train_ratio"],
        val_ratio=args["val_ratio"],
        test_ratio=args["test_ratio"],
        seed=args["split_seed"],
    )
    args["train_mask"] = train_mask.to(args["device"])
    args["val_mask"] = val_mask.to(args["device"])
    args["test_mask"] = test_mask.to(args["device"])
    client_node_indices = partition_data(data, args["num_clients"])
    secure_pretrainer = pretrainer_cls(data, args, data.x, client_node_indices)
    global_aggregated_features = secure_pretrainer.perform_secure_aggregation()
    input_dim = global_aggregated_features.shape[1]
    labels = data.y.to(args["device"])
    clients = [
        Client(i, client_node_indices[i], global_aggregated_features, labels, args, num_classes)
        for i in range(args["num_clients"])
    ]
    global_model = SimpleLinearModel(input_dim, num_classes).to(args["device"])
    for round_idx in range(args["global_rounds"]):
        global_weights = global_model.state_dict()
        local_weights = []
        local_sizes = []
        for client in clients:
            client.update_weights(copy.deepcopy(global_weights))
            weights, size = client.train()
            local_weights.append(copy.deepcopy(weights))
            local_sizes.append(size)
        updated_weights = copy.deepcopy(global_weights)
        total_size = sum(local_sizes)
        for key in updated_weights.keys():
            updated_weights[key] = torch.zeros_like(updated_weights[key])
            for i in range(len(local_weights)):
                updated_weights[key] += local_weights[i][key] * (local_sizes[i] / total_size)
        global_model.load_state_dict(updated_weights)
        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            global_model.eval()
            with torch.no_grad():
                val_acc_server = server_evaluate(global_model, global_aggregated_features, labels, args["val_mask"])
                test_acc_server = server_evaluate(global_model, global_aggregated_features, labels, args["test_mask"])
                total_correct, total_test = 0, 0
                for client in clients:
                    local_test_mask = args["test_mask"][client.node_indices]
                    if local_test_mask.sum() == 0:
                        continue
                    correct, size = client.evaluate(local_test_mask)
                    total_correct += correct
                    total_test += size
                test_acc_client = total_correct / total_test if total_test > 0 else 0.0
                print(
                    f"Round {round_idx+1:03d} | "
                    f"Val Acc (Server): {val_acc_server:.4f} | "
                    f"Test Acc (Server): {test_acc_server:.4f} | "
                    f"Test Acc (Client Avg): {test_acc_client:.4f}"
                )

