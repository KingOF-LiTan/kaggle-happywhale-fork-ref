import pandas as pd
import ast
from pathlib import Path

ROOT = Path("happywhale_data")
PSEUDO_PATH = ROOT / "pseudo_labels/round1_body.csv"
BACKFIN_PATH = ROOT / "test_backfin.csv" # 对应 dataset.py 里的默认 backfin 文件

def parse_bbox(x):
    try:
        if not isinstance(x, str): return []
        # bbox 格式通常是 "[xmin, ymin, xmax, ymax]" 或 "[[xmin, ymin, xmax, ymax]]"
        # dataset.py 里是: lambda x: [list(map(int, x[2:-2].split()))] ...
        # 这里为了通用，尝试 ast.literal_eval
        # 但 dataset.py 那个 split 写法暗示格式可能是 "[ [x y w h] ]" 这种空格分隔？
        # 让我们看 dataset.py line 49: df_bbox_backfin = pd.read_csv(...)
        # line 86: df["bbox_backfin"] = df_bbox_backfin["bbox"].map(
        #     lambda x: [list(map(int, x[2:-2].split()))] if isinstance(x, str) else []
        # )
        # x[2:-2] 去掉两头括号，然后 split。说明是字符串。
        # 能够 split 说明里面是空格分隔的数字？
        # 比如 "[[10 10 50 50]]" -> "10 10 50 50" -> [10, 10, 50, 50]
        # 或者 "[10, 10, 50, 50]" -> ...
        # 我们直接根据 dataset.py 的逻辑来 mock：
        if x == "[]" or x == "": return []
        # 假设格式是 "[[x, y, w, h]]" 或类似。
        # 简单判定：长度 < 5 肯定没数据
        if len(x) < 5: return []
        return [1] # 只要不空就行
    except:
        return []

def main():
    if not PSEUDO_PATH.exists():
        print(f"File not found: {PSEUDO_PATH}")
        return

    print("Loading pseudo labels...")
    df_pseudo = pd.read_csv(PSEUDO_PATH)
    print(f"Total pseudo labels: {len(df_pseudo)}")

    print(f"Loading backfin bboxes from {BACKFIN_PATH}...")
    df_backfin = pd.read_csv(BACKFIN_PATH)
    
    # Merge
    merged = pd.merge(df_pseudo, df_backfin[['image', 'bbox']], on='image', how='left')
    
    # Check bbox validity
    # 模拟 dataset.py 的解析逻辑
    def is_valid_bbox(x):
        if not isinstance(x, str): return False
        x = x.strip()
        if x == '' or x == '[]': return False
        # 粗略检查：是否包含数字
        return any(c.isdigit() for c in x)

    merged['has_bbox'] = merged['bbox'].apply(is_valid_bbox)
    
    n_valid = merged['has_bbox'].sum()
    n_total = len(merged)
    
    print(f"\n--- Statistics ---")
    print(f"Total Pseudo Labels: {n_total}")
    print(f"With Valid Backfin Bbox: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"Missing Backfin Bbox: {n_total - n_valid} ({(n_total - n_valid)/n_total*100:.1f}%)")
    
    if n_total - n_valid > 0:
        print("\n[WARNING] Found pseudo labels without backfin bboxes!")
        print("These samples will fallback to FULL IMAGE in training, causing domain shift for Fin model.")
        print("Suggestion: Filter out these samples in generate_pseudo_labels.py")

if __name__ == "__main__":
    main()
