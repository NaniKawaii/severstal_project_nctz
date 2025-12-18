from __future__ import annotations
import os
import json
import matplotlib.pyplot as plt


def plot_history(history_path: str, out_dir: str, best_epoch: int | None = None):
    """
    Genera gráficas:
    - Training vs Validation Loss
    - Training vs Validation Dice

    Si best_epoch es provisto (1-indexed), marca:
    - línea vertical punteada
    - punto del mejor valor de validación
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(history_path, "r", encoding="utf-8") as f:
        h = json.load(f)

    epochs = list(range(1, len(h["train_loss"]) + 1))

    # =======================
    # LOSS CURVE
    # =======================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, h["train_loss"], label="Training")
    plt.plot(epochs, h["val_loss"], label="Validation")

    if best_epoch is not None and 1 <= best_epoch <= len(epochs):
        plt.axvline(best_epoch, linestyle="--")
        plt.scatter(best_epoch, h["val_loss"][best_epoch - 1])

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=160)
    plt.close()

    # =======================
    # DICE CURVE
    # =======================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, h["train_dice"], label="Training")
    plt.plot(epochs, h["val_dice"], label="Validation")

    if best_epoch is not None and 1 <= best_epoch <= len(epochs):
        plt.axvline(best_epoch, linestyle="--")
        plt.scatter(best_epoch, h["val_dice"][best_epoch - 1])

    plt.title("Training and Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_curve.png"), dpi=160)
    plt.close()

    # =======================
    # DICE PER CLASS (VALIDATION)
    # =======================
    has_per_class = all(k in h for k in [
        "val_dice_c1", "val_dice_c2", "val_dice_c3", "val_dice_c4"
    ])

    if has_per_class:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, h["val_dice_c1"], label="Class 1")
        plt.plot(epochs, h["val_dice_c2"], label="Class 2")
        plt.plot(epochs, h["val_dice_c3"], label="Class 3")
        plt.plot(epochs, h["val_dice_c4"], label="Class 4")

        if best_epoch is not None and 1 <= best_epoch <= len(epochs):
            plt.axvline(best_epoch, linestyle="--")

        plt.title("Validation Dice per Class")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dice_per_class.png"), dpi=160)
        plt.close()
