import os
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from tqdm import tqdm

def parse_conf(x):
    try:
        if isinstance(x, str):
            if x == '[]': return 0.0
            val = ast.literal_eval(x)
            if isinstance(val, list):
                while isinstance(val, list) and len(val) > 0:
                    val = val[0]
                return float(val) if not isinstance(val, list) else 0.0
        elif isinstance(x, (int, float)):
            return float(x)
        return 0.0
    except:
        return 0.0

def analyze_anomalies(csv_path, output_report):
    df = pd.read_csv(csv_path)
    
    # 这里的 bbox 格式假设是 [[xmin ymin xmax ymax]] 或 []
    # 统计 num_box
    def count_boxes(x):
        try:
            if not isinstance(x, str) or x == '[]': return 0
            val = ast.literal_eval(x)
            return len(val) if isinstance(val, list) else 0
        except: return 0

    df['num_box'] = df['bbox'].apply(count_boxes)
    df['parsed_conf'] = df['conf'].apply(parse_conf)
    
    # 筛选异常
    # 1. 无框 (num_box == 0)
    no_box = df[df['num_box'] == 0]
    
    # 2. 多框 (num_box > 1)
    multi_box = df[df['num_box'] > 1]
    
    # 3. 低分 (score < 0.4)
    low_score = df[(df['num_box'] == 1) & (df['parsed_conf'] < 0.4)]
    
    anomalies = pd.concat([no_box, multi_box, low_score]).drop_duplicates().copy()
    
    print(f"Analysis for {csv_path}:")
    print(f" - Total images: {len(df)}")
    print(f" - No box: {len(no_box)}")
    print(f" - Multi box: {len(multi_box)}")
    print(f" - Low score (<0.4): {len(low_score)}")
    print(f" - Total anomalies: {len(anomalies)} ({len(anomalies)/len(df):.2%})")
    
    anomalies.to_csv(output_report, index=False)
    print(f"Anomaly report saved to {output_report}")

if __name__ == "__main__":
    # 分析训练集 backfin
    analyze_anomalies(
        "happywhale_data/train_backfin_ensembled.csv", 
        "happywhale_data/train_anomalies_report.csv"
    )
