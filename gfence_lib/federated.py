import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

class Client:
    def __init__(self, client_id, node_indices, aggregated_features, labels, args, num_classes):
        self.id = client_id
        self.node_indices = node_indices.to(args["device"])
        self.train_mask = args["train_mask"][self.node_indices]
        self.features = aggregated_features[self.node_indices].detach()
        self.labels = labels[self.node_indices]
        self.args = args
        self.model = SimpleLinearModel(self.features.shape[1], num_classes).to(args["device"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
        self.criterion = nn.CrossEntropyLoss()

    def update_weights(self, global_state_dict):
        self.model.load_state_dict(global_state_dict)

    def train(self):
        self.model.train()
        for _ in range(self.args["local_epochs"]):
            self.optimizer.zero_grad()
            output = self.model(self.features)
            loss = self.criterion(output[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            self.optimizer.step()
        return self.model.state_dict(), len(self.node_indices)

    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        output = self.model(self.features)
        pred = output.argmax(dim=1)
        correct = (pred[mask] == self.labels[mask]).sum().item()
        return correct, mask.sum().item()

@torch.no_grad()
def server_evaluate(model, features, labels, mask):
    model.eval()
    out = model(features)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item() if mask.sum().item() > 0 else 0.0

