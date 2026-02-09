import sys
import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import cv2
from omegaconf import OmegaConf

# 1. 确保能找到项目根目录下的 run 模块
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from run.pl_model import PLModel
from run.init.preprocessing import Preprocessing
from src.datasets.wrapper import WrapperDataset

def l2_normalize(x):
    # ... (保持原样)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

@torch.inference_mode()
def extract_feature_from_crop(img, model, preprocessing, device):
    transform = preprocessing.get_test_transform()
    # 单张 crop 的预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img_rgb)["image"]
    batch = transformed.unsqueeze(0).to(device)

    # 通过模型的 forward_features 提取特征
    features = model.forwarder.model.forward_features(batch)
    return features.cpu().numpy()

def refine_bboxes(multi_box_json, train_embed_path, output_csv, model_cfg_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load model and preprocessing
    print("Loading model for feature extraction...")
    
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # 指向真实的配置目录：kaggle-happywhale-fork-ref/run/conf
    config_dir = "../run/conf"
    config_name = Path(model_cfg_path).stem # 获取 config_effb0
        
    with initialize(version_base=None, config_path=config_dir):
        # 1. 先加载主配置
        cfg = compose(config_name=config_name)
        
        # 2. 手动加载 dataset 配置组以解决插值失败问题
        dataset_yaml = root_dir / "run/conf/dataset/happy_whale.yaml"
        if dataset_yaml.exists():
            print(f"Manually merging dataset config from {dataset_yaml}")
            # 重要：解除结构限制，允许注入 dataset 下的字段
            OmegaConf.set_struct(cfg, False)
            dataset_cfg = OmegaConf.load(dataset_yaml)
            cfg = OmegaConf.merge(cfg, {"dataset": dataset_cfg})
        else:
            print(f"Warning: dataset config not found at {dataset_yaml}")
    
    model = PLModel(cfg)
    # ... (后续加载 ckpt 逻辑保持不变)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    
    preprocessing = Preprocessing(cfg.augmentation, **cfg.preprocessing)
    
    # 2. Load reference train embeddings
    print("Loading reference train embeddings...")
    train_data = np.load(train_embed_path)
    train_feat = l2_normalize(train_data["embed_features1"])
    
    # 3. Load multi-box inference results
    with open(multi_box_json, 'r') as f:
        multi_boxes = json.load(f)
    
    images_dir = Path("happywhale_data/test_images")
    refined_results = []

    print("Refining bboxes based on similarity...")
    for img_name, boxes in tqdm(multi_boxes.items()):
        img_path = images_dir / img_name
        if not img_path.exists() or len(boxes) == 0:
            refined_results.append({"image": img_name, "bbox": "[]", "conf": "[]"})
            continue
            
        if len(boxes) == 1:
            box = boxes[0]
            bbox_str = f"[[{int(box['bbox'][0])} {int(box['bbox'][1])} {int(box['bbox'][2])} {int(box['bbox'][3])}]]"
            conf_str = f"[{box['conf']:.4f}]"
            refined_results.append({"image": img_name, "bbox": bbox_str, "conf": conf_str})
            continue

        # Handle multiple boxes
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            refined_results.append({"image": img_name, "bbox": "[]", "conf": "[]"})
            continue
            
        box_features = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box["bbox"])
            crop = img_bgr[max(0, ymin):ymax, max(0, xmin):xmax]
            if crop.size == 0:
                feat = np.zeros((1, train_feat.shape[1]))
            else:
                feat = extract_feature_from_crop(crop, model, preprocessing, device)
            box_features.append(l2_normalize(feat))
            
        box_features = np.vstack(box_features) # (num_boxes, dim)
        
        # Calculate cosine distances to all train samples
        # distance = 1 - cosine_similarity
        distances = 1.0 - np.dot(box_features, train_feat.T) # (num_boxes, num_train)
        min_distances = np.min(distances, axis=1) # (num_boxes,)
        
        best_idx = 0
        min_dist = np.min(min_distances)
        
        if min_dist > 0.5:
            # If distance > 0.5, choose the box with overall minimum distance
            best_idx = np.argmin(min_distances)
        else:
            # If distance < 0.5, choose the box with highest detection score
            # Filter boxes that have dist < 0.5
            valid_indices = np.where(min_distances < 0.5)[0]
            scores = [boxes[i]["conf"] for i in valid_indices]
            best_idx = valid_indices[np.argmax(scores)]
            
        best_box = boxes[best_idx]
        bbox_str = f"[[{int(best_box['bbox'][0])} {int(best_box['bbox'][1])} {int(best_box['bbox'][2])} {int(best_box['bbox'][3])}]]"
        conf_str = f"[{best_box['conf']:.4f}]"
        refined_results.append({"image": img_name, "bbox": bbox_str, "conf": conf_str})

    # Save to CSV
    df_refined = pd.DataFrame(refined_results)
    # Merge with original sample_submission to keep all images
    sub_df = pd.read_csv("happywhale_data/sample_submission.csv")
    final_df = pd.merge(sub_df[['image']], df_refined, on='image', how='left')
    final_df.to_csv(output_csv, index=False)
    print(f"Saved refined bboxes to {output_csv}")

if __name__ == "__main__":
    # This script requires:
    # 1. test_multi_boxes.json (from yolo_inference_multi_boxes.py)
    # 2. concat_train.npz (or body_train.npz)
    # 3. A trained model ckpt for feature extraction
    
    refine_bboxes(
        multi_box_json="happywhale_data/test_multi_boxes.json",
        train_embed_path="/root/autodl-tmp/kaggle-happywhale-fork-ref/outputs/emb/body_ref_train.npz",
        output_csv="happywhale_data/test_backfin_refined.csv",
        model_cfg_path="run/conf/config_effb0.yaml", # Use the same cfg as the ckpt
        ckpt_path="/root/autodl-tmp/kaggle-happywhale-fork-ref/outputs/body_b0_for_refinement/checkpoints/last.ckpt" # Update to your best body ckpt
    )
