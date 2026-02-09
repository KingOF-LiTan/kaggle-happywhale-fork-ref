import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import argparse

def retry_missing_boxes(model_path, anomaly_report_csv, output_json, imgsz=1024, conf=0.001):
    model = YOLO(model_path)
    df_anomaly = pd.read_csv(anomaly_report_csv)
    # Only retry those with no box
    missing_images = df_anomaly[df_anomaly['bbox'] == '[]']['image'].tolist()
    
    if not missing_images:
        print("No missing boxes found in report.")
        return

    images_dir = Path("happywhale_data/test_images")
    retry_results = {}

    print(f"Retrying {len(missing_images)} images with imgsz={imgsz}, conf={conf}...")
    for img_name in tqdm(missing_images):
        img_path = images_dir / img_name
        if not img_path.exists(): continue
        
        results = model(str(img_path), verbose=False, conf=conf, imgsz=imgsz)
        
        img_boxes = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                c = float(box.conf.cpu().numpy()[0])
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                img_boxes.append({"bbox": xyxy, "conf": c})
        
        retry_results[img_name] = img_boxes

    with open(output_json, 'w') as f:
        json.dump(retry_results, f)
    print(f"Retry results saved to {output_json}")

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    MODEL_PATH = ROOT_DIR / "runs" / "detect" / "outputs" / "yolo_backfin" / "v1_n_6404" / "weights" / "best.pt"
    
    retry_missing_boxes(
        model_path=str(MODEL_PATH),
        anomaly_report_csv="happywhale_data/refinement_anomaly_report.csv",
        output_json="happywhale_data/test_multi_boxes_retry.json"
    )
