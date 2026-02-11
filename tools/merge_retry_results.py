import json
import argparse
from pathlib import Path

def merge_results(phase):
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = ROOT_DIR / "happywhale_data"
    
    base_json_path = DATA_DIR / f"{phase}_multi_boxes.json"
    retry_json_path = DATA_DIR / f"{phase}_multi_boxes_retry.json"
    output_json_path = DATA_DIR / f"{phase}_multi_boxes_merged.json"
    
    if not base_json_path.exists():
        print(f"Error: Base JSON {base_json_path} not found.")
        return
    if not retry_json_path.exists():
        print(f"Error: Retry JSON {retry_json_path} not found. Did you run the retry script?")
        return

    with open(base_json_path, 'r') as f:
        base_data = json.load(f)
    with open(retry_json_path, 'r') as f:
        retry_data = json.load(f)

    print(f"Merging {len(retry_data)} retried samples into {len(base_data)} base samples for {phase}...")
    
    merge_count = 0
    for img_name, boxes in retry_data.items():
        if img_name in base_data:
            # If the base had no boxes, replace with retry results
            if not base_data[img_name] and boxes:
                base_data[img_name] = boxes
                merge_count += 1
            # Optional: if you want to append instead of replace, use base_data[img_name].extend(boxes)
    
    with open(output_json_path, 'w') as f:
        json.dump(base_data, f)
    
    print(f"Successfully merged {merge_count} samples. Output saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True, choices=["train", "test"])
    args = parser.parse_args()
    merge_results(args.phase)
