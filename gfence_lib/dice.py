import torch

class DiceEnsembleSampler:
    def __init__(self, target_pmf, support_values, T, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_pmf = target_pmf.to(self.device, dtype=torch.float64)
        self.support_values = support_values.to(self.device)
        self.T = int(T)
        self.n_support = len(target_pmf)
        self.tables = None

    def build_tables(self, n_layers):
        curr_dist = self.target_pmf.clone()
        self.tables = torch.full((n_layers, self.T), -1, dtype=torch.long, device=self.device)
        support_indices = torch.arange(self.n_support, device=self.device)
        for layer in range(n_layers):
            blocks = torch.floor(curr_dist * self.T).long()
            filled_elements = torch.repeat_interleave(support_indices, blocks)
            n_filled = len(filled_elements)
            if n_filled > self.T:
                filled_elements = filled_elements[:self.T]
                n_filled = self.T
            self.tables[layer, :n_filled] = filled_elements
            captured_prob = blocks.float() / self.T
            curr_dist -= captured_prob
            remaining_total = curr_dist.sum()
            if remaining_total < 1e-15:
                break
            curr_dist /= remaining_total

    def sample_noise_one_hot_logic(self, n_samples):
        if self.tables is None:
            raise ValueError("Run build_tables() first.")
        n_layers = self.tables.shape[0]
        final_sample_indices = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        active_mask = torch.ones(n_samples, dtype=torch.bool, device=self.device)
        for layer in range(n_layers):
            if not active_mask.any():
                break
            random_indices = torch.randint(0, self.T, (n_samples,), device=self.device)
            layer_values = self.tables[layer][random_indices]
            is_valid = layer_values != -1
            update_mask = active_mask & is_valid
            final_sample_indices[update_mask] = layer_values[update_mask]
            active_mask = active_mask & (~is_valid)
        if active_mask.any():
            final_sample_indices[active_mask] = self.n_support // 2
        return self.support_values[final_sample_indices]

def generate_dg_props(sigma, r=2**20, L=2**19, device="cuda"):
    x_half = torch.arange(1, r + 1, dtype=torch.float64, device=device)
    probs_half = torch.exp(-(x_half**2) / (2 * sigma**2))
    prob_zero = torch.tensor([1.0], dtype=torch.float64, device=device)
    C = 2 * probs_half.sum() + prob_zero
    x_target = torch.arange(-L, L + 1, dtype=torch.float64, device=device)
    probs_target = torch.exp(-(x_target**2) / (2 * sigma**2)) / C
    probs_target /= probs_target.sum()
    return probs_target, x_target

