import pandas as pd
import ast
import numpy as np
from pathlib import Path

def get_stats(path):
    if not Path(path).exists():
        return None
    
    df = pd.read_csv(path)
    
    def parse_conf(x):
        try:
            if isinstance(x, str):
                if x == '[]':
                    return 0.0
                # Handle formats like [0.9321] or [[0.9321]]
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

    confs = df['conf'].apply(parse_conf)
    
    mean_conf = confs.mean()
    det_rate_05 = (confs > 0.5).mean()
    total_det_rate = (confs > 0.0).mean()
    
    return {
        "mean_conf": mean_conf,
        "det_rate_05": det_rate_05,
        "total_det_rate": total_det_rate,
        "count": len(df)
    }

files = {
    "Original": "happywhale_data/train_backfin.csv",
    "Charm": "happywhale_data/backfin_train_charm.csv",
    "YOLOv8n (Self)": "happywhale_data/train_backfin_yolov8.csv"
}

print(f"{'Source':<15} | {'Mean Conf':<10} | {'Det Rate>0.5':<12} | {'Total Det':<10}")
print("-" * 60)

for name, path in files.items():
    stats = get_stats(path)
    if stats:
        print(f"{name:<15} | {stats['mean_conf']:<10.4f} | {stats['det_rate_05']:<12.2%} | {stats['total_det_rate']:<10.2%}")
    else:
        print(f"{name:<15} | File not found")
