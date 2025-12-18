from __future__ import annotations
import torch

@torch.no_grad()
def dice_per_class(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    pred/target: (B, C, H, W) as float {0,1}
    returns: (B, C)
    """
    num = 2.0 * (pred * target).sum(dim=(2,3))
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return num / den

@torch.no_grad()
def mean_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    return dice_per_class(pred, target).mean().item()
