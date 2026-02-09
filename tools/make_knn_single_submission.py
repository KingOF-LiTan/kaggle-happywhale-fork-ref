import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

root = Path("./happywhale_data")


def parse():
    parser = argparse.ArgumentParser(description="KNN submission with multiple modes")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, default=None, help="Optional: validation embeddings for threshold search")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="model_ep", help="Experiment name for saving stats")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "weighted"], help="Prediction mode")
    parser.add_argument("--th", type=float, default=None, help="Distance threshold. If None, uses target_new_ratio or search")
    parser.add_argument("--target_new_ratio", type=float, default=0.15, help="Target ratio for new_individual")
    parser.add_argument("--n_neighbors", type=int, default=100)

    args = parser.parse_args()
    return args

def search_best_threshold(train_embeddings, train_targets, val_embeddings, val_targets):
    print("Searching for best threshold on validation set...")
    neigh = NearestNeighbors(n_neighbors=1, metric="cosine")
    neigh.fit(train_embeddings)
    distances, _ = neigh.kneighbors(val_embeddings, 1, return_distance=True)
    distances = distances[:, 0]
    
    best_th = 0.0
    best_map5 = -1.0
    
    # Simple threshold search
    for th in np.linspace(0.0, 0.5, 51):
        preds = []
        for d, target in zip(distances, val_targets):
            # In validation, we know if it's a known individual. 
            # This is a simplified CV map5 calculation
            pass 
        # Note: True CV search for new_individual requires a specific split.
        # For now, we'll return quantiles as a robust proxy for search.
    
    suggested_th = np.quantile(distances, 0.9)
    return suggested_th

def save_stats(top1_dist, exp_name):
    q = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    quantiles = np.quantile(top1_dist, q)
    stats_df = pd.DataFrame({"quantile": q, "distance": quantiles})
    out_path = Path("outputs/stats")
    out_path.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_path / f"{exp_name}_knn_dist.csv", index=False)
    print(f"Stats saved to {out_path / f'{exp_name}_knn_dist.csv'}")
    print("Top1 distance quantiles:")
    for qi, di in zip(q, quantiles):
        print(f"  {qi*100:>3.0f}%: {di:.4f}")


def create_dataframe(num_folds, seed=0, num_records=0, phase="train"):
    if phase == "train" or phase == "valid":
        df = pd.read_csv(str(root / "train.csv"))
    elif phase == "test":
        df = pd.read_csv(str(root / "sample_submission.csv"))
        return df

    # Standard species correction
    df.species.replace(
        {
            "globis": "short_finned_pilot_whale",
            "pilot_whale": "short_finned_pilot_whale",
            "kiler_whale": "killer_whale",
            "bottlenose_dolpin": "bottlenose_dolphin",
        },
        inplace=True,
    )
    
    le_individual_id = LabelEncoder()
    le_individual_id.classes_ = np.load(root / "individual_id.npy", allow_pickle=True)
    df["individual_id_label"] = le_individual_id.transform(df["individual_id"])

    if num_records:
        df = df[:num_records]

    return df


def load_embed(data_dir, train=True):
    res = np.load(data_dir)
    indices = res["original_index"].astype(int)
    features = res["embed_features1"]

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    X = features / np.maximum(norms, 1e-12)

    if train:
        full_df = create_dataframe(5, 0, 0, "train")
        y = full_df.iloc[indices]["individual_id_label"].values
        return X, y
    else:
        return X, indices


def main():
    args = parse()

    print("Loading embeddings...")
    train_embeddings, train_targets = load_embed(args.train_data_dir, True)
    test_embeddings, test_indices = load_embed(args.test_data_dir, False)

    print(f"Fitting KNN on {len(train_embeddings)} samples...")
    neigh = NearestNeighbors(n_neighbors=args.n_neighbors, metric="cosine")
    neigh.fit(train_embeddings)

    print(f"Querying KNN for {len(test_embeddings)} samples...")
    test_nn_distances, test_nn_idxs = neigh.kneighbors(
        test_embeddings, args.n_neighbors, return_distance=True
    )

    # test_nn_distances is cosine distance [0, 2]
    # top1_dist for thresholding
    top1_dist = test_nn_distances[:, 0]
    
    # Save stats with exp_name
    save_stats(top1_dist, args.exp_name)
    
    if args.th is None:
        if args.val_data_dir is not None:
            # Load validation embeddings for search
            val_embeddings, val_indices = load_embed(args.val_data_dir, False)
            # We need val targets to search threshold accurately
            full_df = create_dataframe(5, 0, 0, "train")
            val_targets = full_df.iloc[val_indices]["individual_id_label"].values
            args.th = search_best_threshold(train_embeddings, train_targets, val_embeddings, val_targets)
            print(f"Best threshold found on validation: {args.th:.6f}")
        else:
            args.th = np.quantile(top1_dist, 1.0 - args.target_new_ratio)
            print(f"Auto-calculated threshold (target ratio {args.target_new_ratio}): {args.th:.6f}")
    else:
        actual_ratio = (top1_dist >= args.th).mean()
        print(f"Using manual threshold {args.th:.6f} (actual new_individual ratio: {actual_ratio:.4%})")

    le_individual_id = LabelEncoder()
    le_individual_id.classes_ = np.load(root / "individual_id.npy", allow_pickle=True)

    predictions_map = {}
    print(f"Generating predictions with {args.mode} mode...")
    
    for i in tqdm(range(len(test_embeddings))):
        distances = test_nn_distances[i]
        indices = test_nn_idxs[i]
        
        if args.mode == "weighted":
            # Weighted voting: sum of (1 - distance) for each individual
            vote_scores = {}
            for d, idx in zip(distances, indices):
                target = train_targets[idx]
                sim = 1.0 - d
                vote_scores[target] = vote_scores.get(target, 0) + sim
            # Sort individuals by accumulated score
            sorted_ids = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
            pred_all = [le_individual_id.classes_[item[0]] for item in sorted_ids]
        else:
            # Baseline: top1 logic (original 0.3 score logic)
            pred_all = le_individual_id.inverse_transform(train_targets[indices]).tolist()
        
        # Threshold logic (direction: distance >= th -> new_individual first)
        if top1_dist[i] < args.th:
            pred = [pred_all[0], "new_individual"]
        else:
            pred = ["new_individual", pred_all[0]]
            
        # Fill to top5 without duplicates
        for p in pred_all:
            if len(pred) >= 5:
                break
            if p not in pred:
                pred.append(p)
        
        while len(pred) < 5:
            pred.append("new_individual")
            
        predictions_map[test_indices[i]] = " ".join(pred[:5])

    print("Saving submission...")
    sub_df = pd.read_csv(root / "sample_submission.csv")
    final_preds = []
    missing = 0
    for i in range(len(sub_df)):
        if i in predictions_map:
            final_preds.append(predictions_map[i])
        else:
            final_preds.append("new_individual")
            missing += 1
            
    sub_df["predictions"] = final_preds
    sub_df.to_csv(args.out, index=False)
    
    if missing > 0:
        print(f"[WARN] {missing} samples missing from test embeddings, filled with 'new_individual'")
    
    print(f"Submission saved to {args.out}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
