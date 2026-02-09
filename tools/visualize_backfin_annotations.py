import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ann_xlsx",
        type=str,
        default="tools/backfin_annotations.xlsx",
    )
    p.add_argument(
        "--images_dir",
        type=str,
        default="happywhale_data/train_images",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="happywhale_data/yolo_backfin/vis",
    )
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thickness", type=int, default=3)
    return p.parse_args()


def _try_parse_single_col(df: pd.DataFrame) -> pd.DataFrame:
    # Current file has a single column named 'filename,x,y,w,h' and values like 'xxx.jpg,0,113,804,512'
    if df.shape[1] == 1:
        col = df.columns[0]
        parts = df[col].astype(str).str.split(",", expand=True)
        if parts.shape[1] >= 5:
            parts = parts.iloc[:, :5]
            parts.columns = ["filename", "x", "y", "w", "h"]
            return parts
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["x", "y", "w", "h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    args = parse_args()
    ann_path = Path(args.ann_xlsx)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(str(ann_path))
    df = _try_parse_single_col(df)

    required = {"filename", "x", "y", "w", "h"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Annotation columns mismatch. Need {sorted(required)}, got {df.columns.tolist()}"
        )

    df = df[list(required)].copy()
    df = _coerce_numeric(df)
    df = df.dropna(subset=["x", "y", "w", "h", "filename"]).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    n = min(args.num, len(df))
    sample_idx = rng.choice(len(df), size=n, replace=False)

    ok = 0
    missing = 0
    invalid = 0

    for j, i in enumerate(sample_idx.tolist()):
        row = df.iloc[i]
        fn = str(row["filename"])
        x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])

        img_path = images_dir / fn
        if not img_path.exists():
            missing += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing += 1
            continue

        H, W = img.shape[:2]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        # clamp
        x1c = max(0, min(W - 1, x1))
        y1c = max(0, min(H - 1, y1))
        x2c = max(0, min(W - 1, x2))
        y2c = max(0, min(H - 1, y2))

        if x2c <= x1c or y2c <= y1c:
            invalid += 1
            continue

        cv2.rectangle(img, (x1c, y1c), (x2c, y2c), (0, 255, 0), args.thickness)
        label = f"{fn} x={x1} y={y1} w={int(w)} h={int(h)}"
        cv2.putText(
            img,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out_path = out_dir / f"{j:04d}_{Path(fn).stem}.jpg"
        cv2.imwrite(str(out_path), img)
        ok += 1

    print(f"Saved {ok} visualizations to: {out_dir}")
    print(f"Missing images: {missing}")
    print(f"Invalid boxes: {invalid}")


if __name__ == "__main__":
    main()
