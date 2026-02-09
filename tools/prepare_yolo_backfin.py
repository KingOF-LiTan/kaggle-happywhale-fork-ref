import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

def prepare_yolo_dataset(ann_path, images_dir, output_root, val_size=0.1, seed=42):
    # 转换为绝对路径
    script_dir = Path(__file__).parent.absolute()
    root_dir = script_dir.parent
    
    ann_path = root_dir / ann_path
    images_dir = root_dir / images_dir
    output_root = root_dir / output_root

    print(f"Using paths:\n - Annotation: {ann_path}\n - Images: {images_dir}\n - Output: {output_root}")

    if not ann_path.exists():
        raise FileNotFoundError(f"Cannot find annotation file at: {ann_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Cannot find images directory at: {images_dir}")

    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    # Read and parse xlsx
    try:
        df = pd.read_excel(ann_path)
    except Exception as e:
        print(f"Error reading excel: {e}")
        # Try different engine if default fails
        df = pd.read_excel(ann_path, engine='openpyxl')

    if df.shape[1] == 1:
        col = df.columns[0]
        df = df[col].astype(str).str.split(",", expand=True)
        df.columns = ["filename", "x", "y", "w", "h"]
    
    for c in ["x", "y", "w", "h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # Split using numpy to avoid sklearn dependency
    np.random.seed(seed)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    val_count_idx = int(len(df) * val_size)
    val_idx = indices[:val_count_idx]
    train_idx = indices[val_count_idx:]
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    def convert_to_yolo(df_split, img_out, lbl_out, phase_name):
        count = 0
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Processing {phase_name}"):
            fn = row["filename"]
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            
            src_path = images_dir / fn
            if not src_path.exists():
                continue
            
            # Get image size
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            # YOLO format: class x_center y_center width height (normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Clip to [0, 1]
            x_center = max(0, min(x_center, 1))
            y_center = max(0, min(y_center, 1))
            norm_w = max(0, min(norm_w, 1))
            norm_h = max(0, min(norm_h, 1))

            # Copy image (using copy2 to preserve metadata)
            shutil.copy2(src_path, img_out / fn)
            
            # Write label
            with open(lbl_out / f"{Path(fn).stem}.txt", "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            count += 1
        return count

    train_count = convert_to_yolo(train_df, images_train, labels_train, "Train")
    val_count = convert_to_yolo(val_df, images_val, labels_val, "Val")

    # Create yaml using absolute path for robustness
    yaml_content = f"""path: {output_root.absolute().as_posix()}
train: images/train
val: images/val

names:
  0: backfin
"""
    with open(output_root / "backfin.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nDone! Summary:")
    print(f" - Train samples: {train_count}")
    print(f" - Val samples: {val_count}")
    print(f" - YAML config: {output_root / 'backfin.yaml'}")

if __name__ == "__main__":
    prepare_yolo_dataset(
        ann_path="tools/backfin_annotations.xlsx",
        images_dir="happywhale_data/train_images",
        output_root="happywhale_data/yolo_backfin"
    )
