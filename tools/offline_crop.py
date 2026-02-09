import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import ast
import argparse

def parse_bbox(bbox_str):
    try:
        if not isinstance(bbox_str, str) or bbox_str == '[]':
            return None
        
        # Remove brackets and split by any whitespace or comma
        s = bbox_str.replace('[', '').replace(']', '').replace(',', ' ').strip()
        parts = [int(float(x)) for x in s.split() if x.strip()]
        
        if len(parts) == 4:
            return parts # [xmin, ymin, xmax, ymax]
    except Exception as e:
        print(f"Error parsing bbox '{bbox_str}': {e}")
        return None
    return None

def offline_crop(csv_path, images_dir, output_dir, target_size=512, margin=0.1):
    df = pd.read_csv(csv_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cropping images from {csv_path}...")
    
    count_ok = 0
    count_fail = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['image']
        bbox = parse_bbox(row['bbox'])
        
        src_path = images_dir / img_name
        dst_path = output_dir / img_name
        
        if not src_path.exists():
            count_fail += 1
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            count_fail += 1
            continue

        h, w = img.shape[:2]
        
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            
            # Apply margin
            dx = xmax - xmin
            dy = ymax - ymin
            xmin = max(0, int(xmin - dx * margin))
            ymin = max(0, int(ymin - dy * margin))
            xmax = min(w, int(xmax + dx * margin))
            ymax = min(h, int(ymax + dy * margin))
            
            crop_img = img[ymin:ymax, xmin:xmax]
            if crop_img.size == 0:
                crop_img = img # Fallback to original if crop fails
        else:
            crop_img = img

        # Resize to target size
        final_img = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Save
        cv2.imwrite(str(dst_path), final_img)
        count_ok += 1

    print(f"Done! Saved: {count_ok}, Failed: {count_fail}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--type", type=str, default="backfin", choices=["backfin", "fb"], help="Crop type: backfin or fb (fullbody)")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    if args.type == "backfin":
        csv_train = "happywhale_data/train_backfin_ensembled.csv"
        csv_test = "happywhale_data/test_backfin_ensembled.csv"
        out_prefix = "backfin"
    else:
        # For fullbody, we use the original or ensembled body csv
        # If you haven't ensembled body yet, we use the authors' version
        csv_train = "happywhale_data/fullbody_train.csv"
        csv_test = "happywhale_data/fullbody_test.csv"
        out_prefix = "fb"

    if args.phase == "train":
        offline_crop(
            csv_path=csv_train,
            images_dir="happywhale_data/train_images",
            output_dir=f"happywhale_data/cropped/{out_prefix}_train_{args.size}",
            target_size=args.size
        )
    else:
        offline_crop(
            csv_path=csv_test,
            images_dir="happywhale_data/test_images",
            output_dir=f"happywhale_data/cropped/{out_prefix}_test_{args.size}",
            target_size=args.size
        )
