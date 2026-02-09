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

def ensemble_csvs(file_paths, output_path):
    dfs = []
    for name, path in file_paths.items():
        if Path(path).exists():
            df = pd.read_csv(path)
            df['src_name'] = name
            df['parsed_conf'] = df['conf'].apply(parse_conf)
            dfs.append(df)
        else:
            print(f"Warning: {path} not found, skipping.")

    if not dfs:
        print("No files to ensemble.")
        return

    # Use the first DF as base for images list
    base_df = dfs[0].copy()
    all_images = base_df['image'].unique()
    
    final_rows = []
    
    # Create a map for quick lookup
    lookup = {name: df.set_index('image') for name, df in zip(file_paths.keys(), dfs)}

    print(f"Ensembling {len(all_images)} images...")
    for img in tqdm(all_images):
        best_conf = -1.0
        best_row = None
        
        for name in file_paths.keys():
            if name in lookup and img in lookup[name].index:
                row = lookup[name].loc[img]
                # If multiple rows for same image (shouldn't happen with our logic), take the first
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                
                conf = row['parsed_conf']
                if conf > best_conf:
                    best_conf = conf
                    best_row = row
        
        if best_row is not None:
            # Clean up temporary columns
            res_dict = best_row.to_dict()
            res_dict.pop('src_name', None)
            res_dict.pop('parsed_conf', None)
            res_dict['image'] = img
            final_rows.append(res_dict)

    df_final = pd.DataFrame(final_rows)
    # Ensure columns match original order as much as possible
    cols = [c for c in base_df.columns if c not in ['src_name', 'parsed_conf']]
    df_final = df_final[cols]
    
    df_final.to_csv(output_path, index=False)
    print(f"Saved ensembled CSV to {output_path}")

if __name__ == "__main__":
    # Train set
    train_files = {
        "Original": "happywhale_data/train_backfin.csv",
        "Charm": "happywhale_data/backfin_train_charm.csv",
        "YOLOv8n": "happywhale_data/train_backfin_yolov8.csv"
    }
    ensemble_csvs(train_files, "happywhale_data/train_backfin_ensembled.csv")

    # Test set
    test_files = {
        "Original": "happywhale_data/test_backfin.csv",
        "Charm": "happywhale_data/backfin_test_charm.csv",
        "YOLOv8n": "happywhale_data/test_backfin_yolov8.csv"
    }
    ensemble_csvs(test_files, "happywhale_data/test_backfin_ensembled.csv")
