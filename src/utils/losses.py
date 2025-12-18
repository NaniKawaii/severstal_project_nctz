from __future__ import annotations
import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.6, dice_weight: float = 0.4):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        eps = 1e-7
        num = 2.0 * (probs * targets).sum(dim=(2,3))
        den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
        dice = (num / den).mean()
        loss = self.bce_w * bce + self.dice_w * (1.0 - dice)
        return loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        eps = 1e-7
        tp = (probs * targets).sum(dim=(2,3))
        fp = (probs * (1 - targets)).sum(dim=(2,3))
        fn = ((1 - probs) * targets).sum(dim=(2,3))
        tversky = (tp + eps) / (tp + self.alpha * fp + self.beta * fn + eps)
        return (1.0 - tversky).mean()
