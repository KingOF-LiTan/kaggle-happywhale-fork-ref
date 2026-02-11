"""
KNN Submission (Concat Mode)
将 Body embedding 和 Fin embedding 拼接成一个长向量后做 KNN 检索。
对于没有 Fin embedding 的样本，使用零填充。

用法:
python tools/make_knn_concat_submission.py \
    --body_train_dir ./outputs/emb/b7_body_train.npz \
    --body_test_dir  ./outputs/emb/b7_body_test.npz \
    --fin_train_dir  ./outputs/emb/b7_fin_train.npz \
    --fin_test_dir   ./outputs/emb/b7_fin_test.npz \
    --val_data_dir   ./outputs/emb/b7_body_val.npz \
    --fin_val_dir    ./outputs/emb/b7_fin_val.npz \
    --out ./outputs/submission_concat_v3.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

root = Path("./happywhale_data")


def parse():
    parser = argparse.ArgumentParser(description="Concat-mode KNN submission")
    parser.add_argument("--body_train_dir", type=str, required=True)
    parser.add_argument("--body_test_dir", type=str, required=True)
    parser.add_argument("--fin_train_dir", type=str, required=True)
    parser.add_argument("--fin_test_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, default=None, help="Body val embeddings")
    parser.add_argument("--fin_val_dir", type=str, default=None, help="Fin val embeddings")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="concat_ep")
    parser.add_argument("--th", type=float, default=None)
    parser.add_argument("--n_neighbors", type=int, default=100)
    parser.add_argument("--fin_weight", type=float, default=0.4,
                        help="Weight for fin features relative to body (default 1.0 = equal)")
    parser.add_argument("--save_concat", type=str, default=None,
                        help="Save concatenated embeddings to this directory for reuse")
    parser.add_argument("--load_concat", type=str, default=None,
                        help="Load pre-saved concat embeddings (skip concatenation)")
    parser.add_argument("--th_only", action="store_true",
                        help="Only search threshold on val, skip test inference")
    return parser.parse_args()


def load_npz(path):
    if path is None:
        return None, None
    p = Path(path)
    if not p.exists():
        print(f"❌ File not found: {p.absolute()}")
        return None, None
    res = np.load(path)
    return res["embed_features1"], res["original_index"].astype(int)


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def create_dataframe():
    df = pd.read_csv(root / "train.csv")
    df.species.replace({
        "globis": "short_finned_pilot_whale",
        "pilot_whale": "short_finned_pilot_whale",
        "kiler_whale": "killer_whale",
        "bottlenose_dolpin": "bottlenose_dolphin",
    }, inplace=True)
    le_id = LabelEncoder()
    le_id.classes_ = np.load(root / "individual_id.npy", allow_pickle=True)
    df["individual_id_label"] = le_id.transform(df["individual_id"])
    return df, le_id


def concat_features(body_feat, body_idx, fin_feat, fin_idx, fin_weight=1.0):
    """将 body 和 fin embedding 按 original_index 对齐后拼接。
    
    - 先 L2 normalize 各自的特征
    - 对于在 body 中有但 fin 中没有的样本，fin 部分用零填充
    - 对于在 fin 中有但 body 中没有的样本，body 部分用零填充
    - 返回: (concat_features, aligned_indices)
    """
    body_feat = norm(body_feat)
    fin_feat = norm(fin_feat)
    dim_body = body_feat.shape[1]
    dim_fin = fin_feat.shape[1]

    # 合并所有出现过的 index
    body_map = {idx: i for i, idx in enumerate(body_idx)}
    fin_map = {idx: i for i, idx in enumerate(fin_idx)}
    all_idx = sorted(set(body_idx.tolist()) | set(fin_idx.tolist()))

    concat = np.zeros((len(all_idx), dim_body + dim_fin), dtype=np.float32)
    for j, idx in enumerate(all_idx):
        if idx in body_map:
            concat[j, :dim_body] = body_feat[body_map[idx]]
        if idx in fin_map:
            concat[j, dim_body:] = fin_feat[fin_map[idx]] * fin_weight

    # Re-normalize the concatenated vector
    concat = norm(concat)
    return concat, np.array(all_idx, dtype=int)


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
                if len(pred) >= 5:
                    break
                if p not in pred:
                    pred.append(p)
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


def main():
    args = parse()
    df_train, le_id = create_dataframe()
    train_targets = df_train["individual_id_label"].values

    # ========== 加载/拼接 embeddings ==========
    if args.load_concat:
        print(f"Loading pre-saved concat embeddings from {args.load_concat}...")
        ldir = Path(args.load_concat)
        tr = np.load(ldir / "concat_train.npz")
        cat_tr_f, cat_tr_idx = tr["features"], tr["indices"].astype(int)
        te = np.load(ldir / "concat_test.npz")
        cat_te_f, cat_te_idx = te["features"], te["indices"].astype(int)
        cat_tr_targets = train_targets[cat_tr_idx]
        print(f"  Train: {cat_tr_f.shape}, Test: {cat_te_f.shape}")
    else:
        print("Loading Body/Fin embeddings...")
        b_tr_f, b_tr_idx = load_npz(args.body_train_dir)
        b_te_f, b_te_idx = load_npz(args.body_test_dir)
        f_tr_f, f_tr_idx = load_npz(args.fin_train_dir)
        f_te_f, f_te_idx = load_npz(args.fin_test_dir)

        for name, feat in [("Body train", b_tr_f), ("Body test", b_te_f),
                            ("Fin train", f_tr_f), ("Fin test", f_te_f)]:
            if feat is None:
                print(f"❌ Critical: {name} features missing.")
                return
            print(f"  {name}: {feat.shape}")

        print(f"\nConcatenating train features (fin_weight={args.fin_weight})...")
        cat_tr_f, cat_tr_idx = concat_features(b_tr_f, b_tr_idx, f_tr_f, f_tr_idx, args.fin_weight)
        cat_tr_targets = train_targets[cat_tr_idx]
        print(f"  Concat train: {cat_tr_f.shape} ({len(cat_tr_idx)} samples)")

        fin_train_set = set(f_tr_idx.tolist())
        body_only_train = sum(1 for idx in b_tr_idx if idx not in fin_train_set)
        print(f"  Body-only (zero-padded fin): {body_only_train}")

        print("Concatenating test features...")
        cat_te_f, cat_te_idx = concat_features(b_te_f, b_te_idx, f_te_f, f_te_idx, args.fin_weight)
        print(f"  Concat test: {cat_te_f.shape} ({len(cat_te_idx)} samples)")

        # 保存拼接后的 embeddings
        if args.save_concat:
            save_dir = Path(args.save_concat)
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez(save_dir / "concat_train.npz", features=cat_tr_f, indices=cat_tr_idx)
            np.savez(save_dir / "concat_test.npz", features=cat_te_f, indices=cat_te_idx)
            print(f"  ✅ Saved concat embeddings to {save_dir}")

    # ========== 建立 KNN Index ==========
    print(f"Building KNN index (n_neighbors={args.n_neighbors})...")
    neigh = NearestNeighbors(n_neighbors=args.n_neighbors, metric="cosine").fit(cat_tr_f)

    # ========== 阈值搜索 ==========
    if args.th is None and args.val_data_dir:
        # 尝试从缓存加载 val
        cat_v_f, cat_v_idx = None, None
        if args.load_concat:
            val_path = Path(args.load_concat) / "concat_val.npz"
            if val_path.exists():
                vd = np.load(val_path)
                cat_v_f, cat_v_idx = vd["features"], vd["indices"].astype(int)
                print(f"Loaded cached val: {cat_v_f.shape}")

        if cat_v_f is None:
            v_f, v_idx = load_npz(args.val_data_dir)
            f_v_f, f_v_idx = load_npz(args.fin_val_dir)
            if v_f is not None:
                print("\nConcatenating validation features...")
                if f_v_f is not None:
                    cat_v_f, cat_v_idx = concat_features(v_f, v_idx, f_v_f, f_v_idx, args.fin_weight)
                else:
                    dim_fin = cat_tr_f.shape[1] // 2
                    v_f_normed = norm(v_f)
                    cat_v_f = np.hstack([v_f_normed, np.zeros((len(v_f), dim_fin), dtype=np.float32)])
                    cat_v_f = norm(cat_v_f)
                    cat_v_idx = v_idx

                # 保存 val concat
                if args.save_concat:
                    save_dir = Path(args.save_concat)
                    np.savez(save_dir / "concat_val.npz", features=cat_v_f, indices=cat_v_idx)
                    print(f"  ✅ Saved concat val to {save_dir}")

        if cat_v_f is not None:
            v_targets = train_targets[cat_v_idx]
            print(f"  Concat val: {cat_v_f.shape} ({len(cat_v_idx)} samples)")
            args.th = search_best_threshold(
                cat_tr_f, cat_tr_targets, cat_v_f, v_targets,
                neighbors=min(50, args.n_neighbors),
            )

    if args.th_only:
        print("\n--th_only mode, skipping test inference.")
        return

    # ========== 推理 ==========
    th = args.th if args.th else 0.15
    print(f"\nInference (th={th:.4f})...")
    distances, nbrs = neigh.kneighbors(cat_te_f, args.n_neighbors, return_distance=True)

    predictions_map = {}
    top1_dists = []
    for i, idx in enumerate(tqdm(cat_te_idx)):
        d, n = distances[i], nbrs[i]
        targets = cat_tr_targets[n]
        top1_dists.append(d[0])

        pred_all = le_id.inverse_transform(targets).tolist()
        pred = [pred_all[0], "new_individual"] if d[0] < th else ["new_individual", pred_all[0]]
        for p in pred_all:
            if len(pred) >= 5:
                break
            if p not in pred:
                pred.append(p)
        while len(pred) < 5:
            pred.append("new_individual")
        predictions_map[idx] = " ".join(pred[:5])

    # ========== 保存结果 ==========
    top1_dists = np.array(top1_dists)
    q_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    qs = np.quantile(top1_dists, q_levels)
    out_path = Path("outputs/stats")
    out_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"quantile": q_levels, "distance": qs}).to_csv(
        out_path / f"{args.exp_name}_dist.csv", index=False
    )
    print(f"\nStats: Q50={qs[5]:.4f}, Q90={qs[7]:.4f}, Q95={qs[8]:.4f}")

    sub = pd.read_csv(root / "sample_submission.csv")
    sub["predictions"] = [predictions_map.get(i, "new_individual") for i in range(len(sub))]
    sub.to_csv(args.out, index=False)
    new_ratio = (sub.predictions.str.split().str[0] == "new_individual").mean()
    print(f"Done. Threshold: {th:.4f}")
    print(f"Submission saved to {args.out}. New ratio: {new_ratio:.4%}")


if __name__ == "__main__":
    main()
