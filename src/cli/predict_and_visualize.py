from __future__ import annotations
import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from src.models.model_factory import build_model
from src.data.dataset import SeverstalSegDataset
from src.utils.config import load_yaml

# Colores por clase (RGB)
DEFECT_COLORS = {
    0: (255, 0, 0),    # Defect 1 - Rojo
    1: (0, 255, 0),    # Defect 2 - Verde
    2: (0, 0, 255),    # Defect 3 - Azul
    3: (255, 255, 0),  # Defect 4 - Amarillo
}


def overlay_mask(image, mask, alpha=0.5):
    """Superpone máscara binaria sobre la imagen."""
    overlay = image.copy()
    for c in range(4):
        color = DEFECT_COLORS[c]
        m = mask[c] > 0
        overlay[m] = (overlay[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_images", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (usamos TEST para visualizar)
    test_df = torch.load(os.path.join(args.splits_dir, "test_df.pt")) \
        if os.path.exists(os.path.join(args.splits_dir, "test_df.pt")) \
        else None

    if test_df is None:
        import pandas as pd
        test_df = pd.read_csv(os.path.join(args.splits_dir, "test.csv"))

    H, W = cfg.get("image_size", [256, 1024])
    images_dir = os.path.join(args.data_dir, "train_images")

    ds = SeverstalSegDataset(
        test_df.head(args.num_images),
        images_dir,
        (H, W),
        train=False
    )

    # Modelo
    model = build_model(cfg.get("model", {})).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    out_pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(out_pred_dir, exist_ok=True)

    for idx in range(len(ds)):
        x, _ = ds[idx]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        pred_mask = (probs > args.threshold).astype(np.uint8)

        # Cargar imagen original
        img_path = os.path.join(images_dir, ds.df.iloc[idx]["ImageId"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))

        overlay = overlay_mask(img, pred_mask)

        # ---- Plot final ----
        plt.figure(figsize=(14, 4))
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.title(f"Original: {ds.df.iloc[idx]['ImageId']}")
        plt.axis("off")

        plt.subplot(2, 1, 2)
        plt.imshow(overlay)
        plt.title("Predictions")
        plt.axis("off")

        save_path = os.path.join(out_pred_dir, f"pred_{idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

        print(f"✅ Guardado: {save_path}")


if __name__ == "__main__":
    main()
