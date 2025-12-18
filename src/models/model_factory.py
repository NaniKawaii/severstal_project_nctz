from __future__ import annotations
from typing import Dict, Any
import segmentation_models_pytorch as smp
import torch.nn as nn

def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    arch = model_cfg.get("arch", "unet").lower()
    encoder = model_cfg.get("encoder", "efficientnet_b0")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    in_channels = int(model_cfg.get("in_channels", 3))
    classes = int(model_cfg.get("classes", 4))

    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif arch == "fpn":
        return smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")
