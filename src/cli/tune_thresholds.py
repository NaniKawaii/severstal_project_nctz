from __future__ import annotations
import os, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.model_factory import build_model
from src.data.dataset import SeverstalSegDataset
from src.utils.metrics import dice_per_class
from src.utils.config import load_yaml
from tqdm import tqdm

@torch.no_grad()
def eval_thresholds(model, loader, device, thresholds):
    model.eval()
    all_scores = []
    thr = torch.tensor(thresholds, device=device).view(1, -1, 1, 1)
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        probs = torch.sigmoid(model(x))
        pred = (probs > thr).float()
        all_scores.append(dice_per_class(pred, y).cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)  # (N,4)
    return scores.mean(axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--grid", default="0.20,0.30,0.40,0.50,0.60,0.70")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = cfg.get("image_size", [256,1024])
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))

    val_df = pd.read_csv(os.path.join(args.splits_dir, "val.csv"))
    images_dir = os.path.join(args.data_dir, "train_images")
    val_ds = SeverstalSegDataset(val_df, images_dir, (H,W), train=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = build_model(cfg.get("model", {})).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    grid = [float(x.strip()) for x in args.grid.split(",") if x.strip()]
    best_thr = [0.5,0.5,0.5,0.5]
    best_score = [-1,-1,-1,-1]

    # tune each class independently
    for c in range(4):
        for t in grid:
            thr = best_thr.copy()
            thr[c] = t
            score_c = eval_thresholds(model, val_loader, device, thr)[c]
            if score_c > best_score[c]:
                best_score[c] = float(score_c)
                best_thr[c] = float(t)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"thresholds": best_thr, "val_dice_per_class": best_score}
    with open(os.path.join(args.out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("✅ Thresholds guardados en", os.path.join(args.out_dir, "thresholds.json"))
    print(out)

if __name__ == "__main__":
    main()
