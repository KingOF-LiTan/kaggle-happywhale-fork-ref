import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def run_multi_box_inference(model_path, images_dir, image_list=None):
    model = YOLO(model_path)
    images_dir = Path(images_dir)
    
    if image_list is None:
        image_list = [f.name for f in images_dir.glob("*.jpg")]
    
    results_map = {}
    print(f"Running multi-box inference on {len(image_list)} images from {images_dir}...")
    
    for img_name in tqdm(image_list):
        img_path = images_dir / img_name
        if not img_path.exists(): continue
        
        # Get all boxes with low threshold
        results = model(str(img_path), verbose=False, conf=0.1)
        
        img_boxes = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf.cpu().numpy()[0])
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                img_boxes.append({
                    "bbox": xyxy, # [xmin, ymin, xmax, ymax]
                    "conf": conf
                })
        
        results_map[img_name] = img_boxes

    return results_map

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    MODEL_PATH = ROOT_DIR / "runs" / "detect" / "outputs" / "yolo_backfin" / "v1_n_6404" / "weights" / "best.pt"
    
    # 1. 训练集
    train_results = run_multi_box_inference(
        model_path=str(MODEL_PATH),
        images_dir=ROOT_DIR / "happywhale_data" / "train_images"
    )
    with open(ROOT_DIR / "happywhale_data" / "train_multi_boxes.json", 'w') as f:
        json.dump(train_results, f)

    # 2. 测试集
    test_results = run_multi_box_inference(
        model_path=str(MODEL_PATH),
        images_dir=ROOT_DIR / "happywhale_data" / "test_images"
    )
    with open(ROOT_DIR / "happywhale_data" / "test_multi_boxes.json", 'w') as f:
        json.dump(test_results, f)
    
    print("Done! Generated train_multi_boxes.json and test_multi_boxes.json")