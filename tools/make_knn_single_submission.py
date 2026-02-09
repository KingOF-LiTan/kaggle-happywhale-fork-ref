import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

root = Path("./happywhale_data")

def parse():
    parser = argparse.ArgumentParser(description="Advanced Dual-Path KNN submission")
    parser.add_argument("--body_train_dir", type=str, required=True)
    parser.add_argument("--body_test_dir", type=str, required=True)
    parser.add_argument("--fin_train_dir", type=str, default=None)
    parser.add_argument("--fin_test_dir", type=str, default=None)
    parser.add_argument("--val_data_dir", type=str, default=None, help="Validation embeddings for TH search")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="model_ep", help="Name for stats saving")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "weighted"])
    parser.add_argument("--th", type=float, default=None)
    parser.add_argument("--target_new_ratio", type=float, default=0.15)
    parser.add_argument("--n_neighbors", type=int, default=100)
    return parser.parse_args()

def load_npz_data(path):
    if path is None or not Path(path).exists(): return None, None
    res = np.load(path)
    return res["embed_features1"], res["original_index"].astype(int)

def create_dataframe():
    df = pd.read_csv(root / "train.csv")
    df.species.replace({
        "globis": "short_finned_pilot_whale", "pilot_whale": "short_finned_pilot_whale",
        "kiler_whale": "killer_whale", "bottlenose_dolpin": "bottlenose_dolphin"
    }, inplace=True)
    le_id = LabelEncoder()
    le_id.classes_ = np.load(root / "individual_id.npy", allow_pickle=True)
    df["individual_id_label"] = le_id.transform(df["individual_id"])
    return df, le_id

def search_best_threshold(train_feat, train_targets, val_feat, val_targets, neighbors=50):
    print("Searching for best threshold on validation set (MAP@5)...")
    neigh = NearestNeighbors(n_neighbors=neighbors, metric="cosine").fit(train_feat)
    distances, idxs = neigh.kneighbors(val_feat, neighbors, return_distance=True)
    
    best_th, best_map5 = 0.0, -1.0
    for th in np.linspace(0.05, 0.45, 41):
        scores = []
        for i in range(len(val_feat)):
            d, idx, target = distances[i], idxs[i], val_targets[i]
            pred_all = train_targets[idx].tolist()
            pred = [pred_all[0], -1] if d[0] < th else [-1, pred_all[0]]
            for p in pred_all:
                if len(pred) >= 5: break
                if p not in pred: pred.append(p)
            try:
                rank = pred[:5].index(target)
                scores.append(1.0 / (rank + 1))
            except ValueError:
                scores.append(0.0)
        cur_map5 = np.mean(scores)
        if cur_map5 > best_map5:
            best_map5, best_th = cur_map5, th
    print(f"Optimal TH: {best_th:.4f} | Val MAP@5: {best_map5:.4f}")
    return best_th

def save_stats(top1_dist, exp_name):
    q_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    qs = np.quantile(top1_dist, q_levels)
    out_path = Path("outputs/stats")
    out_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"quantile": q_levels, "distance": qs}).to_csv(out_path / f"{exp_name}_dist.csv", index=False)
    print(f"Stats saved. Q90: {qs[7]:.4f}, Q95: {qs[8]:.4f}")

def main():
    args = parse()
    df_train, le_id = create_dataframe()
    train_targets = df_train["individual_id_label"].values

    print("Loading Body/Fin embeddings...")
    b_tr_f, b_tr_idx = load_npz_data(args.body_train_dir)
    b_te_f, b_te_idx = load_npz_data(args.body_test_dir)
    f_tr_f, f_tr_idx = load_npz_data(args.fin_train_dir)
    f_te_f, f_te_idx = load_npz_data(args.fin_test_dir)

    def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    b_tr_f, b_te_f = norm(b_tr_f), norm(b_te_f)
    
    neigh_body = NearestNeighbors(n_neighbors=args.n_neighbors, metric="cosine").fit(b_tr_f)
    fin_test_map = {idx: i for i, idx in enumerate(f_te_idx)} if f_te_idx is not None else {}
    
    if f_tr_f is not None:
        neigh_fin = NearestNeighbors(n_neighbors=args.n_neighbors, metric="cosine").fit(norm(f_tr_f))
    else:
        neigh_fin = None

    if args.th is None and args.val_data_dir:
        v_f, v_idx = load_npz_data(args.val_data_dir)
        v_targets = df_train.iloc[v_idx]["individual_id_label"].values
        args.th = search_best_threshold(b_tr_f, train_targets[b_tr_idx], norm(v_f), v_targets)

    predictions_map = {}
    top1_dists = []
    
    print("Inference with fallback logic...")
    for i, idx in enumerate(tqdm(b_te_idx)):
        # Fallback logic: use Fin if available, else Body
        if neigh_fin and idx in fin_test_map:
            feat = norm(f_te_f[fin_test_map[idx]:fin_test_map[idx]+1])
            d, n = neigh_fin.kneighbors(feat, args.n_neighbors)
            targets = train_targets[f_tr_idx[n[0]]]
        else:
            d, n = neigh_body.kneighbors(b_te_f[i:i+1], args.n_neighbors)
            targets = train_targets[b_tr_idx[n[0]]]
        
        d, n = d[0], n[0]
        top1_dists.append(d[0])
        
        if args.mode == "weighted":
            vote = {}
            for dist, target in zip(d, targets):
                vote[target] = vote.get(target, 0) + (1.0 - dist)
            sorted_ids = sorted(vote.items(), key=lambda x: x[1], reverse=True)
            pred_all = [le_id.classes_[item[0]] for item in sorted_ids]
        else:
            pred_all = le_id.inverse_transform(targets).tolist()

        th = args.th if args.th else np.quantile(top1_dists, 1-args.target_new_ratio)
        pred = [pred_all[0], "new_individual"] if d[0] < th else ["new_individual", pred_all[0]]
        for p in pred_all:
            if len(pred) >= 5: break
            if p not in pred: pred.append(p)
        while len(pred) < 5: pred.append("new_individual")
        predictions_map[idx] = " ".join(pred[:5])

    save_stats(np.array(top1_dists), args.exp_name)
    sub = pd.read_csv(root / "sample_submission.csv")
    sub["predictions"] = [predictions_map.get(i, "new_individual") for i in range(len(sub))]
    sub.to_csv(args.out, index=False)
    print(f"Submission saved to {args.out}. New ratio: {(sub.predictions.str.split().str[0]=='new_individual').mean():.4%}")

if __name__ == "__main__":
    main()
