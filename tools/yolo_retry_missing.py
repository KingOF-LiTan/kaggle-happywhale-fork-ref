import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import argparse

<<<<<<< HEAD
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
        
=======
def retry_missing_boxes(model_path, anomaly_report_csv, output_json, images_dir, imgsz=1024, conf=0.001):
    model = YOLO(model_path)
    anomaly_report_csv = Path(anomaly_report_csv)
    if not anomaly_report_csv.exists():
        print(f"Error: {anomaly_report_csv} not found.")
        return

    df_anomaly = pd.read_csv(anomaly_report_csv)
    # Only retry those with no box (bbox == '[]')
    # Depending on how the report was generated, bbox might be empty string or '[]'
    missing_images = df_anomaly[df_anomaly['bbox'].astype(str) == '[]']['image'].tolist()
    
    if not missing_images:
        print(f"No missing boxes found in {anomaly_report_csv}.")
        return

    images_dir = Path(images_dir)
    retry_results = {}

    print(f"Retrying {len(missing_images)} images from {images_dir} with imgsz={imgsz}, conf={conf}...")
    for img_name in tqdm(missing_images):
        img_path = images_dir / img_name
        if not img_path.exists(): 
            continue
        
        # High resolution retry
>>>>>>> 9e087a31f532a9dbb9d08160cb98c36551b43bfc
        results = model(str(img_path), verbose=False, conf=conf, imgsz=imgsz)
        
        img_boxes = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                c = float(box.conf.cpu().numpy()[0])
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                img_boxes.append({"bbox": xyxy, "conf": c})
        
        retry_results[img_name] = img_boxes

<<<<<<< HEAD
=======
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
>>>>>>> 9e087a31f532a9dbb9d08160cb98c36551b43bfc
    with open(output_json, 'w') as f:
        json.dump(retry_results, f)
    print(f"Retry results saved to {output_json}")

if __name__ == "__main__":
<<<<<<< HEAD
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    MODEL_PATH = ROOT_DIR / "runs" / "detect" / "outputs" / "yolo_backfin" / "v1_n_6404" / "weights" / "best.pt"
    
    retry_missing_boxes(
        model_path=str(MODEL_PATH),
        anomaly_report_csv="happywhale_data/refinement_anomaly_report.csv",
        output_json="happywhale_data/test_multi_boxes_retry.json"
=======
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()

    ROOT_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = ROOT_DIR / "happywhale_data"
    
    # Path to your best YOLO model
    MODEL_PATH = ROOT_DIR / "runs" / "detect" / "outputs" / "yolo_backfin" / "v1_n_6404" / "weights" / "best.pt"
    if not MODEL_PATH.exists():
        MODEL_PATH = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

    report_csv = DATA_DIR / f"{args.phase}_refinement_anomaly_report.csv"
    out_json = DATA_DIR / f"{args.phase}_multi_boxes_retry.json"
    img_dir = DATA_DIR / f"{args.phase}_images"

    retry_missing_boxes(
        model_path=str(MODEL_PATH),
        anomaly_report_csv=report_csv,
        output_json=out_json,
        images_dir=img_dir,
        imgsz=args.imgsz,
        conf=args.conf
>>>>>>> 9e087a31f532a9dbb9d08160cb98c36551b43bfc
    )
