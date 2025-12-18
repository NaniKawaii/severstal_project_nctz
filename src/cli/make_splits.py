from __future__ import annotations
import os
import argparse
from sklearn.model_selection import train_test_split
from src.data.dataset import pivot_train_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    os.makedirs(args.out_dir, exist_ok=True)
    pv = pivot_train_csv(os.path.join(args.data_dir, "train.csv"))

    train_df, temp_df = train_test_split(
        pv, test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=pv["has_defect"]
    )

    # val + test split from temp
    val_size = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_size),
        random_state=args.seed,
        stratify=temp_df["has_defect"]
    )

    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    print("✅ Splits creados en:", args.out_dir)
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

if __name__ == "__main__":
    main()
