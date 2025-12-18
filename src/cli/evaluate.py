from __future__ import annotations
import os, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.model_factory import build_model
from src.data.dataset import SeverstalSegDataset
from src.training.trainer import run_eval
from src.utils.config import load_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--thresholds", default=None, help="JSON con thresholds por clase")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = cfg.get("image_size", [256,1024])
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))

    test_df = pd.read_csv(os.path.join(args.splits_dir, "test.csv"))
    images_dir = os.path.join(args.data_dir, "train_images")
    test_ds = SeverstalSegDataset(test_df, images_dir, (H,W), train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = build_model(cfg.get("model", {})).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    thr = float(cfg.get("thresholds", {}).get("default", 0.5))
    if args.thresholds:
        with open(args.thresholds, "r", encoding="utf-8") as f:
            thr = np.array(json.load(f)["thresholds"], dtype=float)

    metrics = run_eval(model, test_loader, device=device, threshold=thr)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Test metrics guardadas en", os.path.join(args.out_dir, "test_metrics.json"))
    print(metrics)

if __name__ == "__main__":
    main()
