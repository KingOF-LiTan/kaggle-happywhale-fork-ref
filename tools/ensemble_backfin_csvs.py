import pandas as pd
import ast
import numpy as np
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

def ensemble_csvs(file_paths, output_path, missing_species=None):
    if missing_species is None:
        missing_species = []
    
    dfs = []
    for name, path in file_paths.items():
        if Path(path).exists():
            df = pd.read_csv(path)
            df['src_name'] = name
            df['parsed_conf'] = df['conf'].apply(parse_conf)
            dfs.append(df)
        else:
            print(f"Warning: {path} not found, skipping.")

    if not dfs: return

    # Load species metadata from train.csv to handle missing species
    train_meta = pd.read_csv("happywhale_data/train.csv")[['image', 'species']]
    species_map = train_meta.set_index('image')['species'].to_dict()

    base_df = dfs[0].copy()
    all_images = base_df['image'].unique()
    lookup = {name: df.set_index('image') for name, df in zip(file_paths.keys(), dfs)}

    final_rows = []
    print(f"Ensembling {len(all_images)} images with species-aware logic...")
    for img in tqdm(all_images):
        species = species_map.get(img, "unknown")
        best_conf = -1.0
        best_row = None
        
        # 1. If species is missing in our training, prefer Original/Charm directly
        if species in missing_species:
            for name in ["Original", "Charm"]:
                if name in lookup and img in lookup[name].index:
                    row = lookup[name].loc[img]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    conf = row['parsed_conf']
                    if conf > best_conf:
                        best_conf = conf
                        best_row = row
        
        # 2. Otherwise (or if fallback failed), use standard highest confidence
        if best_row is None:
            for name in file_paths.keys():
                if name in lookup and img in lookup[name].index:
                    row = lookup[name].loc[img]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    conf = row['parsed_conf']
                    if conf > best_conf:
                        best_conf = conf
                        best_row = row
        
        if best_row is not None:
            res_dict = best_row.to_dict()
            res_dict.pop('src_name', None)
            res_dict.pop('parsed_conf', None)
            res_dict['image'] = img
            final_rows.append(res_dict)

    df_final = pd.DataFrame(final_rows)
    cols = [c for c in base_df.columns if c not in ['src_name', 'parsed_conf']]
    df_final[cols].to_csv(output_path, index=False)
    print(f"Saved species-aware ensembled CSV to {output_path}")

if __name__ == "__main__":
    MISSING = ['gray_whale', 'beluga', 'southern_right_whale']
    # Train set
    train_files = {
        "Original": "happywhale_data/train_backfin.csv",
        "Charm": "happywhale_data/backfin_train_charm.csv",
        "YOLOv8n": "happywhale_data/train_backfin_yolov8.csv"
    }
    ensemble_csvs(train_files, "happywhale_data/train_backfin_ensembled.csv", missing_species=MISSING)
    # Test set (Assume similar missing species distribution in test)
    test_files = {
        "Original": "happywhale_data/test_backfin.csv",
        "Charm": "happywhale_data/backfin_test_charm.csv",
        "YOLOv8n": "happywhale_data/test_backfin_yolov8.csv"
    }
    ensemble_csvs(test_files, "happywhale_data/test_backfin_ensembled.csv", missing_species=MISSING)
