import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import json

def run_multi_box_inference(model_path, images_dir, output_json):
    model = YOLO(model_path)
    images_dir = Path(images_dir)
    image_list = [f.name for f in images_dir.glob("*.jpg")]
    
    results_map = {}
    print(f"Running multi-box inference on {len(image_list)} images...")
    
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

    with open(output_json, 'w') as f:
        json.dump(results_map, f)
    print(f"Saved multi-box results to {output_json}")

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    MODEL_PATH = ROOT_DIR / "runs" / "detect" / "outputs" / "yolo_backfin" / "v1_n_6404" / "weights" / "best.pt"
    
    # 针对测试集进行多框导出，准备相似度筛选
    run_multi_box_inference(
        model_path=str(MODEL_PATH),
        images_dir=ROOT_DIR / "happywhale_data" / "test_images",
        output_json=ROOT_DIR / "happywhale_data" / "test_multi_boxes.json"
    )
