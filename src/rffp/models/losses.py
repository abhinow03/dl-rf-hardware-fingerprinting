"""rffp.models.losses — Supervised Contrastive (SupCon) loss + temperature schedule.

`SupervisedContrastiveLoss` pulls embeddings of the SAME device together and pushes different
devices apart on the unit hypersphere (the objective that makes the 128-D metric space
clusterable for open-world discovery). `get_temperature(epoch)` is the annealed temperature
schedule used during training (warm high-temp start -> low-temp sharpening).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_temperature(epoch):
    if epoch < 10:
        return 0.5
    elif epoch < 30:
        return 0.1
    return 0.07


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels, temperature=0.1):
        embeddings = embeddings.float()
        device = embeddings.device
        n = embeddings.shape[0]

        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.t()) & ~self_mask

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = torch.mm(embeddings, embeddings.t()) / temperature
        sim_masked = sim.masked_fill(self_mask, float('-inf'))

        # torch.logsumexp handles -inf correctly — no NaN from -inf * 0
        log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)

        log_prob = sim - log_denom
        log_prob = log_prob.masked_fill(self_mask, 0.0)  # zero diagonal explicitly

        loss = -(log_prob * pos_mask).sum(1)
        n_pos = pos_mask.sum(1).float().clamp(min=1)
        loss = (loss / n_pos).mean()

        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss
