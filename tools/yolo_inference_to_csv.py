import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2

def run_inference(model_path, images_dir, output_csv, phase="train", original_csv=None):
    model = YOLO(model_path)
    images_dir = Path(images_dir)
    
    # Load original data to get all images and metadata
    if original_csv and os.path.exists(original_csv):
        df_orig = pd.read_csv(original_csv)
        image_list = df_orig['image'].tolist()
    else:
        image_list = [f.name for f in images_dir.glob("*.jpg")]
        df_orig = pd.DataFrame({'image': image_list})

    results_data = []
    
    print(f"Running inference on {len(image_list)} images from {images_dir}...")
    
    for img_name in tqdm(image_list):
        img_path = images_dir / img_name
        
        # Default empty values
        bbox_str = "[]"
        conf_str = "[]"
        width, height = 0, 0
        
        if img_path.exists():
            # YOLO inference
            results = model(str(img_path), verbose=False, conf=0.01) # Low threshold to capture more
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Take the highest confidence box
                box = results[0].boxes[0]
                conf = box.conf.cpu().numpy()[0]
                # xyxy format
                xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                
                # Format: [[xmin ymin xmax ymax]]
                bbox_str = f"[[{xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]}]]"
                # Format: [conf]
                conf_str = f"[{conf:.4f}]"
                
                orig_shape = results[0].orig_shape # (h, w)
                height, width = orig_shape[0], orig_shape[1]
            else:
                # Try to get image size even if no box found
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width = img.shape[:2]

        res = {
            'image': img_name,
            'bbox': bbox_str,
            'conf': conf_str,
            'width': width,
            'height': height
        }
        results_data.append(res)

    df_res = pd.DataFrame(results_data)
    
    # Merge with original metadata if available
    if original_csv and os.path.exists(original_csv):
        # Drop columns that we recalculated or don't want from original
        cols_to_keep = [c for c in df_orig.columns if c not in ['bbox', 'conf', 'width', 'height']]
        df_final = pd.merge(df_orig[cols_to_keep], df_res, on='image', how='left')
    else:
        df_final = df_res

    df_final.to_csv(output_csv, index=False)
    print(f"Saved inference results to {output_csv}")

if __name__ == "__main__":
    # 使用 pathlib 自动获取项目根目录，避免手动输入路径和转义问题
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    
    # 路径对齐到你训练出的真实位置 v1_n_6404
    MODEL_PATH = ROOT_DIR / "runs" / "detect" / "outputs" / "yolo_backfin" / "v1_n_6404" / "weights" / "best.pt"

    if not MODEL_PATH.exists():
        print(f"❌ Still cannot find model at: {MODEL_PATH}")
        # Fallback to general runs directory if structure is different
        MODEL_PATH = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

    print(f"✅ Final Model Path: {MODEL_PATH}")
    
    # 确保输出目录存在
    DATA_DIR = ROOT_DIR / "happywhale_data"
    
    # Train set
    run_inference(
        model_path=str(MODEL_PATH),
        images_dir=DATA_DIR / "train_images",
        output_csv=DATA_DIR / "train_backfin_yolov8.csv",
        phase="train",
        original_csv=DATA_DIR / "train.csv"
    )
    
    # Test set
    run_inference(
        model_path=str(MODEL_PATH),
        images_dir=DATA_DIR / "test_images",
        output_csv=DATA_DIR / "test_backfin_yolov8.csv",
        phase="test",
        original_csv=DATA_DIR / "sample_submission.csv"
    )
