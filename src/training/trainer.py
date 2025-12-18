from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import dice_per_class
from src.utils.losses import BCEDiceLoss, TverskyLoss

def make_loss(loss_cfg: Dict[str, Any]) -> nn.Module:
    name = str(loss_cfg.get("name", "bce_dice")).lower()
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.6)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.4)),
        )
    if name == "tversky":
        return TverskyLoss(
            alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
            beta=float(loss_cfg.get("tversky_beta", 0.7)),
        )
    raise ValueError(f"Loss no soportada: {name}")

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: str, threshold: float | np.ndarray = 0.5):
    model.eval()
    dices = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        if isinstance(threshold, np.ndarray):
            thr = torch.tensor(threshold, device=device).view(1, -1, 1, 1)
            pred = (probs > thr).float()
        else:
            pred = (probs > float(threshold)).float()

        dices.append(dice_per_class(pred, y).cpu().numpy())
    dices = np.concatenate(dices, axis=0)  # (N,4)
    return {
        "dice_mean": float(dices.mean()),
        "dice_per_class": dices.mean(axis=0).tolist(),
    }

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    out_dir: str,
    device: str,
):
    os.makedirs(out_dir, exist_ok=True)
    loss_fn = make_loss(cfg.get("loss", {})).to(device)

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-4))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best = -1.0
    history = {"train_loss": [], "val_dice": [], "val_dice_c1": [], "val_dice_c2": [], "val_dice_c3": [], "val_dice_c4": []}

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)

        val_metrics = run_eval(model, val_loader, device=device, threshold=float(cfg.get("thresholds", {}).get("default", 0.5)))
        val_dice = val_metrics["dice_mean"]
        c = val_metrics["dice_per_class"]

        sched.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        history["val_dice_c1"].append(c[0])
        history["val_dice_c2"].append(c[1])
        history["val_dice_c3"].append(c[2])
        history["val_dice_c4"].append(c[3])

        # guardar checkpoint
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_mean_dice={val_dice:.4f} | val_per_class={np.round(c,4)}")

    # guardar history
    import json
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history
