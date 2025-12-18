from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.dataset import pivot_train_csv
from src.utils.rle import rle_decode
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Ruta a data/raw/severstal")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_area_samples", type=int, default=2000, help="Muestreo para estimar áreas")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_csv = os.path.join(args.data_dir, "train.csv")
    train_images = os.path.join(args.data_dir, "train_images")

    pv = pivot_train_csv(train_csv)
    pv.to_csv(os.path.join(args.out_dir, "pivot_train.csv"), index=False)

    # stats
    stats = {}
    stats["n_images"] = int(len(pv))
    stats["pct_has_defect"] = float(pv["has_defect"].mean())
    for k in [1,2,3,4]:
        stats[f"pct_class_{k}"] = float(pv[f"rle_{k}"].apply(lambda x: isinstance(x,str)).mean())

    pd.Series(stats).to_csv(os.path.join(args.out_dir, "stats.csv"))

    # barplot defect vs no defect
    plt.figure()
    pv["has_defect"].value_counts().plot(kind="bar")
    plt.title("Conteo: con defecto vs sin defecto")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "has_defect_bar.png"), dpi=160)
    plt.close()

    # class presence
    cls_counts = [pv[f"rle_{k}"].apply(lambda x: isinstance(x,str)).sum() for k in [1,2,3,4]]
    plt.figure()
    plt.bar(["c1","c2","c3","c4"], cls_counts)
    plt.title("Imágenes con presencia por clase")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "class_presence.png"), dpi=160)
    plt.close()

    # rough area distribution (sampled)
    sample = pv.sample(n=min(args.max_area_samples, len(pv)), random_state=42)
    areas = {k: [] for k in [1,2,3,4]}

    # Need one image size; infer from first existing image
    first_img = cv2.imread(os.path.join(train_images, sample.iloc[0]["ImageId"]))
    H, W = first_img.shape[:2]

    for _, row in sample.iterrows():
        for k in [1,2,3,4]:
            rle = row.get(f"rle_{k}")
            if isinstance(rle, str):
                m = rle_decode(rle, H, W)
                areas[k].append(int(m.sum()))

    # plot areas
    plt.figure(figsize=(10,4))
    for k in [1,2,3,4]:
        if len(areas[k]) > 0:
            plt.hist(areas[k], bins=30, alpha=0.6, label=f"c{k}")
    plt.title("Distribución aproximada de áreas de máscara (muestreo)")
    plt.xlabel("Pixeles positivos")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mask_area_hist.png"), dpi=160)
    plt.close()

    print("✅ EDA listo. Revisa:", args.out_dir)

if __name__ == "__main__":
    main()
