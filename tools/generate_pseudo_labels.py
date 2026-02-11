"""
伪标签生成脚本 (v2)

支持两种模式：
1. 单模型模式: 直接传 --train_emb / --test_emb
2. Concat 模式: 传 --load_concat 加载已拼接的 embedding 缓存

输出: happywhale_data/pseudo_labels/{exp_name}.csv
格式: image, individual_id, species (兼容 HappyWhaleData.load_csv)

同时输出去泄漏信息: happywhale_data/pseudo_labels/{exp_name}_leaked_val.npy

用法 (Step 1 - Body only, 高阈值):
  python tools/generate_pseudo_labels.py \
      --train_emb ./outputs/emb/body_train.npz \
      --test_emb ./outputs/emb/body_test.npz \
      --val_emb ./outputs/emb/body_val.npz \
      --sim_threshold 0.7 \
      --exp_name round1_body

用法 (Step 2 - Concat, 低阈值):
  python tools/generate_pseudo_labels.py \
      --load_concat ./outputs/emb/concat_b7_fw05 \
      --val_emb ./outputs/emb/body_val.npz \
      --fin_val_emb ./outputs/emb/fin_val.npz \
      --sim_threshold 0.5 \
      --exp_name round2_concat
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

root = Path("./happywhale_data")


def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_npz(path):
    if path is None:
        return None, None
    p = Path(path)
    if not p.exists():
        print(f"❌ File not found: {p.absolute()}")
        return None, None
    res = np.load(path)
    return res["embed_features1"], res["original_index"].astype(int)


def concat_features(body_feat, body_idx, fin_feat, fin_idx, fin_weight=1.0):
    """Same concat logic as make_knn_concat_submission.py"""
    body_feat = norm(body_feat)
    fin_feat = norm(fin_feat)
    dim_body = body_feat.shape[1]
    dim_fin = fin_feat.shape[1]

    body_map = {idx: i for i, idx in enumerate(body_idx)}
    fin_map = {idx: i for i, idx in enumerate(fin_idx)}
    all_idx = sorted(set(body_idx.tolist()) | set(fin_idx.tolist()))

    concat = np.zeros((len(all_idx), dim_body + dim_fin), dtype=np.float32)
    for j, idx in enumerate(all_idx):
        if idx in body_map:
            concat[j, :dim_body] = body_feat[body_map[idx]]
        if idx in fin_map:
            concat[j, dim_body:] = fin_feat[fin_map[idx]] * fin_weight
    return norm(concat), np.array(all_idx, dtype=int)


def generate_pseudo_labels(args):
    # ========== 加载训练数据和标签 ==========
    df_train = pd.read_csv(root / "train.csv")
    df_train.species.replace({
        "globis": "short_finned_pilot_whale",
        "pilot_whale": "short_finned_pilot_whale",
        "kiler_whale": "killer_whale",
        "bottlenose_dolpin": "bottlenose_dolphin",
    }, inplace=True)

    le_id = LabelEncoder()
    le_id.classes_ = np.load(root / "individual_id.npy", allow_pickle=True)
    df_train["individual_id_label"] = le_id.transform(df_train["individual_id"])
    train_targets = df_train["individual_id_label"].values

    df_test = pd.read_csv(root / "sample_submission.csv")

    # ========== 加载 Embeddings ==========
    if args.load_concat:
        print(f"Loading concat embeddings from {args.load_concat}...")
        ldir = Path(args.load_concat)
        tr = np.load(ldir / "concat_train.npz")
        train_feat, train_idx = tr["features"], tr["indices"].astype(int)
        te = np.load(ldir / "concat_test.npz")
        test_feat, test_idx = te["features"], te["indices"].astype(int)
    else:
        print("Loading single-model embeddings...")
        train_feat, train_idx = load_npz(args.train_emb)
        test_feat, test_idx = load_npz(args.test_emb)
        if train_feat is None or test_feat is None:
            print("❌ Missing embeddings. Abort.")
            return
        train_feat = norm(train_feat)
        test_feat = norm(test_feat)

    print(f"  Train: {train_feat.shape}, Test: {test_feat.shape}")
    train_label_mapped = train_targets[train_idx]

    # ========== KNN 搜索 ==========
    print("Building KNN index...")
    n_neighbors = min(5, len(train_feat))
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(train_feat)
    distances, indices = neigh.kneighbors(test_feat, n_neighbors, return_distance=True)

    # cosine distance → similarity: sim = 1 - dist
    sim_threshold = args.sim_threshold
    dist_threshold = 1.0 - sim_threshold

    print(f"\nGenerating pseudo labels (sim > {sim_threshold}, dist < {dist_threshold:.4f})...")

    # Distance distribution
    top1_dists = distances[:, 0]
    print(f"  Top1 distance distribution:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"    Q{int(q*100):02d}: {np.quantile(top1_dists, q):.4f}")

    # ========== 生成伪标签 ==========
    pseudo_rows = []
    for i in range(len(test_feat)):
        d = distances[i, 0]
        nn_idx = indices[i, 0]
        if d < dist_threshold:
            orig_test_idx = test_idx[i]
            img_name = df_test.iloc[orig_test_idx]["image"]
            predicted_id = le_id.classes_[train_label_mapped[nn_idx]]
            # 获取最近邻训练样本的物种
            nn_train_orig_idx = train_idx[nn_idx]
            species = df_train.iloc[nn_train_orig_idx]["species"]

            pseudo_rows.append({
                "image": img_name,
                "individual_id": predicted_id,
                "species": species,
                "conf": 1.0 - d,  # similarity score, used by dataset.py for filtering
            })

    df_pseudo = pd.DataFrame(pseudo_rows)
    n_total = len(test_feat)
    n_pseudo = len(df_pseudo)
    print(f"\n  Pseudo labels: {n_pseudo}/{n_total} ({n_pseudo/n_total*100:.1f}%)")
    if n_pseudo > 0:
        print(f"  Unique individuals: {df_pseudo['individual_id'].nunique()}")
        print(f"  Species distribution (top 5):")
        for sp, cnt in df_pseudo["species"].value_counts().head(5).items():
            print(f"    {sp}: {cnt}")

    # ========== 去泄漏 ==========
    leaked_val_indices = np.array([], dtype=int)
    if args.val_emb and n_pseudo > 0:
        print("\n  Checking validation set leakage...")

        # 加载 val embeddings
        if args.load_concat:
            val_path = Path(args.load_concat) / "concat_val.npz"
            if val_path.exists():
                vd = np.load(val_path)
                val_feat, val_idx = vd["features"], vd["indices"].astype(int)
            else:
                val_feat, val_idx = load_npz(args.val_emb)
                if val_feat is not None:
                    val_feat = norm(val_feat)
        else:
            val_feat, val_idx = load_npz(args.val_emb)
            if val_feat is not None:
                val_feat = norm(val_feat)

        if val_feat is not None:
            # 获取被选为伪标签的 test embedding 子集
            pseudo_test_orig_indices = [df_test[df_test["image"] == r["image"]].index[0] for r in pseudo_rows]
            pseudo_positions = [np.where(test_idx == idx)[0][0] for idx in pseudo_test_orig_indices]
            pseudo_feat = test_feat[pseudo_positions]

            # 批量计算: 对每个 val 样本，找在 pseudo set 中的最近邻
            print(f"    Computing pseudo({len(pseudo_feat)}) × val({len(val_feat)}) similarity...")
            neigh_pseudo = NearestNeighbors(n_neighbors=1, metric="cosine").fit(pseudo_feat)
            v_dists, _ = neigh_pseudo.kneighbors(val_feat, 1, return_distance=True)
            v_sims = 1.0 - v_dists[:, 0]

            leak_th = args.leak_threshold
            leak_mask = v_sims > leak_th
            leaked_val_indices = val_idx[leak_mask]

            print(f"    Leak threshold: sim > {leak_th}")
            print(f"    Leaked val samples: {len(leaked_val_indices)}/{len(val_feat)} ({len(leaked_val_indices)/len(val_feat)*100:.2f}%)")
            print(f"    Val sim distribution: min={v_sims.min():.4f}, max={v_sims.max():.4f}, "
                  f"mean={v_sims.mean():.4f}, >0.9={np.sum(v_sims>0.9)}")

    # ========== 检查有效背鳍 Bbox (Test Backfin Check) ==========
    # 这一步非常关键：若伪标签样本没有有效的 backfin bbox，
    # Fin 模型训练时会 fallback 到全图，导致严重的 domain shift 和噪声。
    backfin_path = root / "test_backfin.csv"
    if backfin_path.exists():
        print(f"\n  Checking backfin bbox validity against {backfin_path}...")
        df_bbox = pd.read_csv(backfin_path)
        
        # 提取 bbox 并判断是否有效
        # 格式通常是字符串 "[[x, y, w, h]]" 或类似，需解析
        def has_valid_bbox(x):
            try:
                if not isinstance(x, str): return False
                # 简单启发式检查：长度足够且包含数字
                if len(x) < 5: return False
                if x.strip() == "[]": return False
                return True
            except:
                return False
        
        df_bbox["has_valid_backfin"] = df_bbox["bbox"].apply(has_valid_bbox)
        
        # Merge with pseudo labels
        # 注意：df_pseudo 里的 image 列对应 test_backfin.csv 的 image 列
        df_merged = df_pseudo.merge(df_bbox[["image", "has_valid_backfin"]], on="image", how="left")
        
        # 统计
        n_invalid = len(df_merged) - df_merged["has_valid_backfin"].sum()
        if n_invalid > 0:
            print(f"    ⚠️ FOUND {int(n_invalid)} pseudo labels with INVALID/EMPTY backfin bboxes!")
            print(f"    These samples would cause Fin model to train on full images (noise).")
            print(f"    Filtering them out...")
            
            # 过滤
            df_pseudo = df_merged[df_merged["has_valid_backfin"] == True].drop(columns=["has_valid_backfin"])
            print(f"    Remaining pseudo labels: {len(df_pseudo)}")
        else:
            print("    ✅ All pseudo labels have valid backfin bboxes.")
            
    # ========== 保存 ==========
    out_dir = root / "pseudo_labels"
    out_dir.mkdir(exist_ok=True)

    out_csv = out_dir / f"{args.exp_name}.csv"
    df_pseudo.to_csv(out_csv, index=False)
    print(f"\n✅ Pseudo labels saved to {out_csv}")

    if len(leaked_val_indices) > 0:
        leak_path = out_dir / f"{args.exp_name}_leaked_val.npy"
        np.save(leak_path, leaked_val_indices)
        print(f"✅ Leaked val indices saved to {leak_path}")

    # 保存元数据
    meta = {
        "exp_name": args.exp_name,
        "sim_threshold": args.sim_threshold,
        "n_pseudo": len(df_pseudo),
        "n_test": n_total,
        "n_leaked_val": len(leaked_val_indices),
        "concat_mode": args.load_concat is not None,
    }
    meta_path = out_dir / f"{args.exp_name}_meta.txt"
    with open(meta_path, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
    print(f"✅ Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo labels for Happywhale")
    parser.add_argument("--train_emb", type=str, default=None, help="Train embedding npz (single model)")
    parser.add_argument("--test_emb", type=str, default=None, help="Test embedding npz (single model)")
    parser.add_argument("--val_emb", type=str, default=None, help="Val embedding npz (for de-leakage)")
    parser.add_argument("--fin_val_emb", type=str, default=None, help="Fin val embedding npz (concat de-leakage)")
    parser.add_argument("--load_concat", type=str, default=None, help="Load pre-saved concat embeddings dir")
    parser.add_argument("--fin_weight", type=float, default=0.5, help="Fin weight for concat (if not using load_concat)")
    parser.add_argument("--sim_threshold", type=float, default=0.7,
                        help="Similarity threshold (default 0.7 for step1, 0.5 for step2)")
    parser.add_argument("--leak_threshold", type=float, default=0.95,
                        help="Similarity threshold for val leakage detection (default 0.95)")
    parser.add_argument("--exp_name", type=str, default="round1", help="Experiment name for output files")
    args = parser.parse_args()

    if args.load_concat is None and (args.train_emb is None or args.test_emb is None):
        print("❌ Must provide either --load_concat or both --train_emb and --test_emb")
    else:
        generate_pseudo_labels(args)
