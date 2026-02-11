import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import cv2
import argparse
import hydra
from hydra import initialize, compose
from run.pl_model import PLModel
from run.init.preprocessing import Preprocessing
<<<<<<< HEAD

=======
from omegaconf import OmegaConf
>>>>>>> 9e087a31f532a9dbb9d08160cb98c36551b43bfc
def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

@torch.inference_mode()
def extract_feature_from_crop(img, model, preprocessing, device):
    transform = preprocessing.get_test_transform()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img_rgb)["image"]
    batch = transformed.unsqueeze(0).to(device)
    # Correct path to model features in this repo
    features = model.forwarder.model.forward_features(batch)
    return features.cpu().numpy()

def refine_bboxes(phase, train_embed_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = ROOT_DIR / "happywhale_data"
    
    # 1. Config and Model loading
<<<<<<< HEAD
    print(f"Loading full configuration via Hydra for {phase}...")
    with initialize(config_path="../run/conf", version_base=None):
        cfg = compose(config_name="config_effb0")
=======
    print(f"Loading config_effb0.yaml directly for {phase}...")
    # 直接加载 yaml 避免 Hydra 复杂的插值依赖
    cfg = OmegaConf.load("run/conf/config_effb0.yaml")
    dataset_cfg = OmegaConf.load("run/conf/dataset/happy_whale.yaml")
    
    # 重要：解除结构限制，允许注入完整节点
    OmegaConf.set_struct(cfg, False)
    
    # 补全 PLModel 初始化所需的完整 dataset 配置
    cfg.dataset = dataset_cfg
    
    # 显式解析/覆盖模型中的插值字段
    if "model" in cfg and "output_dim_species" in cfg.model:
        cfg.model.output_dim_species = int(dataset_cfg.num_species_classes)
>>>>>>> 9e087a31f532a9dbb9d08160cb98c36551b43bfc
    
    print("Loading model for feature extraction...")
    model = PLModel(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    
    preprocessing = Preprocessing(cfg.augmentation, **cfg.preprocessing)
    
    # 2. Reference embeddings
    print(f"Loading reference train embeddings from {train_embed_path}...")
    train_data = np.load(train_embed_path)
    train_feat = l2_normalize(train_data["embed_features1"])
    
    # 3. Input selection
<<<<<<< HEAD
    multi_box_json = DATA_DIR / f"{phase}_multi_boxes.json"
=======
    multi_box_json = DATA_DIR / f"{phase}_multi_boxes_merged.json"
>>>>>>> 9e087a31f532a9dbb9d08160cb98c36551b43bfc
    images_dir = DATA_DIR / f"{phase}_images"
    output_csv = DATA_DIR / f"{phase}_backfin_refined.csv"
    
    if not multi_box_json.exists():
        raise FileNotFoundError(f"Missing input multi-box JSON: {multi_box_json}")

    with open(multi_box_json, 'r') as f:
        multi_boxes = json.load(f)
    
    refined_results = []

    print(f"Refining bboxes for {phase} based on similarity...")
    for img_name, boxes in tqdm(multi_boxes.items()):
        img_path = images_dir / img_name
        
        # Branch 1: No detection
        if not img_path.exists() or len(boxes) == 0:
            refined_results.append({"image": img_name, "bbox": "[]", "conf": "[]", "min_dist": 1.0})
            continue
            
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            refined_results.append({"image": img_name, "bbox": "[]", "conf": "[]", "min_dist": 1.0})
            continue

        box_features = []
        valid_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box["bbox"])
            # Clamp to image bounds
            h, w = img_bgr.shape[:2]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            
            crop = img_bgr[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            
            feat = extract_feature_from_crop(crop, model, preprocessing, device)
            box_features.append(l2_normalize(feat))
            valid_boxes.append(box)
            
        if not box_features:
            refined_results.append({"image": img_name, "bbox": "[]", "conf": "[]", "min_dist": 1.0})
            continue

        box_features = np.vstack(box_features)
        
        # Similarity measure
        similarities = np.dot(box_features, train_feat.T) 
        max_similarities = np.max(similarities, axis=1) 
        min_distances = 1.0 - max_similarities 
        
        best_idx = 0
        global_min_dist = np.min(min_distances)
        
        # Logic: If only one valid box, use it
        if len(valid_boxes) == 1:
            best_idx = 0
        # Logic: Multi-box decision
        elif global_min_dist > 0.5:
            best_idx = np.argmin(min_distances) # Choose most similar
        else:
            valid_indices = np.where(min_distances < 0.5)[0]
            scores = [valid_boxes[i]["conf"] for i in valid_indices]
            best_idx = valid_indices[np.argmax(scores)] # Choose highest score among similar ones
            
        best_box = valid_boxes[best_idx]
        bbox_str = f"[[{int(best_box['bbox'][0])} {int(best_box['bbox'][1])} {int(best_box['bbox'][2])} {int(best_box['bbox'][3])}]]"
        conf_str = f"[{best_box['conf']:.4f}]"
        refined_results.append({"image": img_name, "bbox": bbox_str, "conf": conf_str, "min_dist": global_min_dist})

    # Save outputs
    df_refined = pd.DataFrame(refined_results)
    
    # Anomaly report
    no_box_samples = df_refined[df_refined['bbox'] == '[]']
    high_dist_samples = df_refined[df_refined['min_dist'] > 0.7]
    report_path = DATA_DIR / f"{phase}_refinement_anomaly_report.csv"
    pd.concat([no_box_samples, high_dist_samples]).to_csv(report_path, index=False)
    print(f"Anomaly report saved to {report_path}")

    # Standard CSV for project
    if phase == "train":
        # Keep metadata if possible
        orig_csv = DATA_DIR / "train.csv"
        if orig_csv.exists():
            df_orig = pd.read_csv(orig_csv)
            # Remove existing columns if merging
            df_orig = df_orig.drop(columns=['bbox', 'conf'], errors='ignore')
            final_df = pd.merge(df_orig, df_refined[['image', 'bbox', 'conf']], on='image', how='left')
        else:
            final_df = df_refined[['image', 'bbox', 'conf']]
    else:
        orig_csv = DATA_DIR / "sample_submission.csv"
        df_orig = pd.read_csv(orig_csv)
        final_df = pd.merge(df_orig[['image']], df_refined[['image', 'bbox', 'conf']], on='image', how='left')

    final_df.to_csv(output_csv, index=False)
    print(f"Final refined CSV saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--train_embed", type=str, default="outputs/emb/body_ref_train.npz")
    parser.add_argument("--ckpt", type=str, default="outputs/body_b0_for_refinement/checkpoints/last.ckpt")
    args = parser.parse_args()

    refine_bboxes(
        phase=args.phase,
        train_embed_path=args.train_embed,
        ckpt_path=args.ckpt
    )
