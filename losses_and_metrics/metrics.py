import torch
from torchmetrics import Metric

"""this files defines the following class:
- AverageCostMetric
"""

class AverageCostMetric(Metric):
    """this class can be used to calculate the AHC 
    for pixel-wise semantic segmentation """
    full_state_update = False

    def __init__(self, D: torch.Tensor, ignore_index=None):
        super().__init__()
        self.register_buffer("D", D)
        self.ignore_index = ignore_index
        self.add_state("total_cost", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    #update states
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        costs = self.D[targets.long(), preds.long()]

        #ignore pixels with real index in ignored_index
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            costs = costs[valid]
        
        #add values to cost and count
        self.total_cost += costs.sum()
        self.count += costs.numel() #amount of pixels

    #calculate AHC
    def compute(self):
        return self.total_cost / self.count