import torch
from torchmetrics import Metric

class AverageCostMetric(Metric):
    full_state_update = False

    def __init__(self, D: torch.Tensor, ignore_index=None):
        super().__init__()
        self.register_buffer("D", D)
        self.ignore_index = ignore_index
        self.add_state("total_cost", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        costs = self.D[targets.long(), preds.long()]
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            costs = costs[valid]
        self.total_cost += costs.sum()
        self.count += costs.numel() #amount of pixels

    def compute(self):
        return self.total_cost / self.count