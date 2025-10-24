import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HierarchicalLoss(nn.Module):
    def __init__(self, distance_matrix: torch.Tensor, ignore_index: int = -1, reduction: str = 'mean'):
        """
        distance_matrix: Tensor of shape (num_classes, num_classes) with D[i, j] = distance(i, j)
        ignore_index: label to ignore in the loss (e.g., for unlabeled pixels)
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.register_buffer("D", distance_matrix.float())
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W) class logits per pixel (B: pct. per batch, C: 62 (nr. classes), H=W=32
        targets: (B, H, W) integer class labels: stores truth class

        """
         
        # Flatten spatial dimensions to compute per-pixel loss
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # change model outputs to probabilities mit e^.. 
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # (B, H, W, C) -> (N, C) N =B*H*C
        targets_flat = targets.reshape(-1)  # 1D vector (N,)
        
        #get rid of invalid indices
        valid_mask = targets_flat != self.ignore_index #should be considered in the loss (T vs. F) #for each pixel individually
        if not torch.any(valid_mask):
            return probs_flat.new_tensor(0.0, requires_grad=True)
        targets_valid = targets_flat[valid_mask] #remove pixels which are not used
        probs_valid = probs_flat[valid_mask]  # (N_valid, C)
        
        
        # calculate the loss:
        D_y = self.D[targets_valid]  # (N_valid, C) for each pixel individual the row in D
        losses = torch.sum(probs_valid * D_y, dim=1)  # (N_valid,), elementwise multiplication (dim 1)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

class ProtoHierarchicalLoss(nn.Module):
    ''' eq. 3'''
    def __init__(self, ignore_index = -1, reduction: str = 'mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, dists: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ''' dists: negative distances for each pixel to all prototypes
        targets: true_labels
        need to be changed - i need outputs for each pixel?
        '''
       
        # Flatten spatial dimensions to compute per-pixel loss
        B, C, H, W = dists.shape
        dists_flat = dists.permute(0, 2, 3, 1).reshape(-1, C)  # (B, H, W, C) -> (N, C) N =B*H*C
        targets_flat = targets.reshape(-1)  # 1D vector (N,)
        
        #get rid of invalid indices
        valid_mask = targets_flat != self.ignore_index #should be considered in the loss (T vs. F) #for each pixel individually
        if not torch.any(valid_mask):
            return dists_flat.new_tensor(0.0, requires_grad=True)
        targets_valid = targets_flat[valid_mask] #remove pixels which are not used
        dists_valid = dists_flat[valid_mask]  # (N_valid, C)
        
        
        # calculate the loss:
        correct_dists = -dists_valid[torch.arange(dists_valid.size(0)), targets_valid]
        log_sum_exp = torch.logsumexp(dists_valid, dim=1)
        losses = correct_dists + log_sum_exp  #1D array per pixel loss
       
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
    




'''see original version: torch_prototypes.metrics.distortion (arXiv:2007.03047)
I only copied it here, so that you dont need to get their code on your devises... '''
class Eucl_Mat(nn.Module):
    """Pairwise Euclidean distance"""

    def __init_(self):
        super(Eucl_Mat, self).__init__()

    def forward(self, mapping):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Euclidean distances

        """
        return torch.norm(mapping[:, None, :] - mapping[None, :, :], dim=-1)

'''see original version: torch_prototypes.metrics.distortion (arXiv:2007.03047)
I only copied it here, so that you dont need to get their code on your devises... '''
class Cosine_Mat(nn.Module):
    """Pairwise Cosine distance"""

    def __init__(self):
        super(Cosine_Mat, self).__init__()

    def forward(self, mapping):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Cosine distances

        """
        return 1 - nn.CosineSimilarity(dim=-1)(mapping[:, None, :], mapping[None, :, :])


'''For DistortionLoss:
    This class was originally implemented by Sainte Fare Garnot, Vivien  and Landrieu, Loic
    I only changed the epsilon in case we have some zeros in our matrix.
    see original version: torch_prototypes.metrics.distortion -> DistortionLoss  (arXiv:2007.03047) '''
class DistortionLoss(nn.Module):
    """Scale-free squared distortion regularizer"""

    def __init__(self, D, dist="euclidian", scale_free=True):
        super(DistortionLoss, self).__init__()
        self.D = D
        self.scale_free = scale_free
        if dist == "euclidian":
            self.dist = Eucl_Mat()
        elif dist == "cosine":
            self.dist = Cosine_Mat()

    def forward(self, mapping, idxs=None):
        d = self.dist(mapping)  
        eps= 1e-8 #changed from me ... also in the denominator below.


        if self.scale_free:
            a = d / (self.D + torch.eye(self.D.shape[0], device=self.D.device) + eps)
            scaling = a.sum() / torch.pow(a, 2).sum()
        else:
            scaling = 1.0

        d = (scaling * d - self.D) ** 2 / (
            self.D + torch.eye(self.D.shape[0], device=self.D.device) + eps) ** 2
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d