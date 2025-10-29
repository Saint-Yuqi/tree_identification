import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossMultiClass(nn.Module):
    """
    Multi-class focal loss for segmentation with ignore_index support.

    Args:
        alpha: Tensor of shape [C] with per-class weights (optional).
        gamma: Focusing parameter gamma >= 0.
        ignore_index: Label to ignore in the target (e.g., -1 or background class id).
        reduction: 'none' | 'mean' | 'sum'.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.register_buffer('alpha', alpha if alpha is not None else None)
        self.gamma = gamma
        self.ignore_index = ignore_index
        assert reduction in {'none', 'mean', 'sum'}
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, C, H, W)
        target: (N, H, W) with values in [0, C-1] or ignore_index
        """
        n, c, h, w = logits.shape
        if target.shape != (n, h, w):
            raise ValueError(f"Target shape must be (N,H,W), got {target.shape} for logits {logits.shape}")

        # Flatten spatial dims for gather
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c)  # (N*H*W, C)
        target_flat = target.reshape(-1)  # (N*H*W)

        # Mask out ignore_index
        valid_mask = target_flat != self.ignore_index
        if valid_mask.sum() == 0:
            # Nothing to compute
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        logits_valid = logits_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        log_prob = F.log_softmax(logits_valid, dim=1)  # (M, C)
        log_pt = log_prob.gather(1, target_valid.unsqueeze(1)).squeeze(1)  # (M,)
        pt = log_pt.exp()

        # Alpha weighting per target class if provided
        if self.alpha is not None:
            if self.alpha.numel() != c:
                raise ValueError(f"alpha length {self.alpha.numel()} must equal num_classes {c}")
            alpha_t = self.alpha[target_valid]
        else:
            alpha_t = 1.0

        focal_term = (1.0 - pt) ** self.gamma
        loss = -alpha_t * focal_term * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # Return spatial shape for compatibility if needed
            out = torch.zeros_like(target_flat, dtype=logits.dtype)
            out[valid_mask] = loss
            return out.view(n, h, w)


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for multi-class segmentation.

    Args:
        effective_alphas: Tensor [C] computed from per-class sample counts via
                          alpha_i = (1 - beta) / (1 - beta ** n_i). Can be normalized.
        gamma: Focal loss focusing parameter.
        ignore_index: label to ignore in the target.
        reduction: reduction method.
    """

    def __init__(self, effective_alphas: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        if effective_alphas is None:
            raise ValueError("effective_alphas must be provided for CBFocalLoss")
        self.focal = FocalLossMultiClass(alpha=effective_alphas, gamma=gamma, ignore_index=ignore_index, reduction=reduction)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, target)

