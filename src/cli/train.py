from __future__ import annotations
import os, argparse, json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.model_factory import build_model
from src.data.dataset import SeverstalSegDataset
from src.training.trainer import train as train_loop
from src.utils.plots import plot_history
from src.utils.config import load_yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_best_epoch(history_path: str) -> int | None:
    """
    Lee history.json y devuelve el epoch (1-indexed) con mejor val_mean_dice.
    Soporta formatos:
    - lista de dicts por epoch
    - dict con listas (train_loss, val_loss, train_dice, val_dice)
    """
    if not os.path.exists(history_path):
        return None

    with open(history_path, "r", encoding="utf-8") as f:
        h = json.load(f)

    # Caso 1: dict con listas
    if isinstance(h, dict):
        # Preferimos val_dice si existe
        if "val_dice" in h and isinstance(h["val_dice"], list) and len(h["val_dice"]) > 0:
            best_idx = int(np.argmax(h["val_dice"]))  # 0-index
            return best_idx + 1
        # Alternativa: val_mean_dice
        if "val_mean_dice" in h and isinstance(h["val_mean_dice"], list) and len(h["val_mean_dice"]) > 0:
            best_idx = int(np.argmax(h["val_mean_dice"]))
            return best_idx + 1
        return None

    # Caso 2: lista de dicts por epoch
    if isinstance(h, list) and len(h) > 0 and isinstance(h[0], dict):
        keys = ["val_mean_dice", "val_dice", "val_dice_mean"]
        best_score = None
        best_epoch = None
        for i, row in enumerate(h, start=1):
            score = None
            for k in keys:
                if k in row:
                    score = row[k]
                    break
            if score is None:
                continue
            if best_score is None or float(score) > float(best_score):
                best_score = float(score)
                best_epoch = i
        return best_epoch

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--loss", default=None, help="Override loss name: bce_dice|tversky")
    ap.add_argument("--encoder", default=None, help="Override encoder: efficientnet_b0, efficientnet_b2, resnet34, ...")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.loss:
        cfg["loss"]["name"] = args.loss
    if args.encoder:
        cfg["model"]["encoder"] = args.encoder
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    H, W = cfg.get("image_size", [256, 1024])
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))

    train_df = pd.read_csv(os.path.join(args.splits_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(args.splits_dir, "val.csv"))

    images_dir = os.path.join(args.data_dir, "train_images")

    train_ds = SeverstalSegDataset(train_df, images_dir, (H, W), train=True,  aug_cfg=cfg.get("augmentation", {}))
    val_ds   = SeverstalSegDataset(val_df,   images_dir, (H, W), train=False)

    # pin_memory solo tiene sentido si hay GPU
    pin_memory = (device == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    model = build_model(cfg.get("model", {})).to(device)

    # Entrenamiento (esto debe guardar history.json y best.pt según tu trainer)
    _ = train_loop(model, train_loader, val_loader, cfg, out_dir=args.out_dir, device=device)

    # plots
    history_path = os.path.join(args.out_dir, "history.json")
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    best_epoch = _find_best_epoch(history_path)
    plot_history(history_path, plots_dir, best_epoch=best_epoch)

    if best_epoch is not None:
        print(f"✅ Gráficas generadas en {plots_dir} (best_epoch={best_epoch})")
    else:
        print(f"✅ Gráficas generadas en {plots_dir} (best_epoch no detectado)")

    print("✅ Entrenamiento completo. Revisa:", args.out_dir)


if __name__ == "__main__":
    main()
