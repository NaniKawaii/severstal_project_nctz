from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.rle import rle_decode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transforms(image_size: Tuple[int, int], train: bool, aug_cfg: Dict[str, Any] | None = None):
    H, W = image_size
    if aug_cfg is None:
        aug_cfg = {}

    if train:
        tfms = [
            A.HorizontalFlip(p=float(aug_cfg.get("hflip_p", 0.5))),
            A.RandomBrightnessContrast(p=float(aug_cfg.get("brightness_contrast_p", 0.3))),
            A.ShiftScaleRotate(
                shift_limit=float(aug_cfg.get("shift_limit", 0.02)),
                scale_limit=float(aug_cfg.get("scale_limit", 0.10)),
                rotate_limit=int(aug_cfg.get("rotate_limit", 5)),
                p=float(aug_cfg.get("geo_p", 0.3)),
                border_mode=cv2.BORDER_REFLECT,
            ),
            A.Resize(height=H, width=W, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        tfms = [
            A.Resize(height=H, width=W, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    return A.Compose(tfms)


def pivot_train_csv(train_csv_path: str) -> pd.DataFrame:
    """
    Crea un dataframe pivot con una fila por ImageId y columnas:
    rle_1, rle_2, rle_3, rle_4 y has_defect.

    Soporta 2 formatos de train.csv:

    A) Formato Kaggle competition:
       - ImageId_ClassId, EncodedPixels
       - Ej: 0002cc93b.jpg_1

    B) Formato ya separado:
       - ImageId, ClassId, EncodedPixels
    """
    df = pd.read_csv(train_csv_path)

    # Limpia espacios accidentales en headers
    df.columns = [c.strip() for c in df.columns]

    if "ImageId_ClassId" in df.columns:
        # Formato A
        df["ImageId"] = df["ImageId_ClassId"].astype(str).apply(lambda x: x.split("_")[0])
        df["ClassId"] = df["ImageId_ClassId"].astype(str).apply(lambda x: int(x.split("_")[1]))
    elif "ImageId" in df.columns and "ClassId" in df.columns:
        # Formato B
        df["ImageId"] = df["ImageId"].astype(str)
        df["ClassId"] = df["ClassId"].astype(int)
    else:
        raise ValueError(
            "train.csv no tiene el formato esperado. "
            "Se esperaba 'ImageId_ClassId' o ('ImageId' y 'ClassId'). "
            f"Columnas encontradas: {list(df.columns)}"
        )

    if "EncodedPixels" not in df.columns:
        raise ValueError(
            "train.csv no contiene la columna 'EncodedPixels'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    pv = (
        df.pivot_table(index="ImageId", columns="ClassId", values="EncodedPixels", aggfunc="first")
          .reset_index()
    )

    # Asegurar que existan columnas 1..4 aunque alguna clase no aparezca
    for c in [1, 2, 3, 4]:
        if c not in pv.columns:
            pv[c] = np.nan

    pv = pv[["ImageId", 1, 2, 3, 4]]
    pv.columns = ["ImageId", "rle_1", "rle_2", "rle_3", "rle_4"]

    pv["has_defect"] = pv[["rle_1", "rle_2", "rle_3", "rle_4"]].apply(
        lambda r: any(isinstance(x, str) for x in r), axis=1
    )
    return pv


class SeverstalSegDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        image_size: Tuple[int, int],
        train: bool,
        aug_cfg: Dict[str, Any] | None = None
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        self.tfms = build_transforms(image_size, train=train, aug_cfg=aug_cfg)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["ImageId"])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        mask = np.zeros((H0, W0, 4), dtype=np.uint8)
        for c in range(4):
            rle = row.get(f"rle_{c+1}")
            if isinstance(rle, str):
                mask[..., c] = rle_decode(rle, H0, W0)

        out = self.tfms(image=img, mask=mask)
        x = out["image"]
        y = out["mask"]

        # Albumentations suele devolver mask como numpy o tensor según ToTensorV2
        if isinstance(y, torch.Tensor):
            # Si viene como HWC, pasar a CHW
            if y.ndim == 3 and y.shape[0] != 4:
                y = y.permute(2, 0, 1)
        else:
            y = torch.from_numpy(y)
            if y.ndim == 3 and y.shape[0] != 4:
                y = y.permute(2, 0, 1)

        return x.float(), y.float()
