import argparse
from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def load_npz(path: str) -> dict:
    res = np.load(path)
    return {k: res[k] for k in res.files}


def build_index_map(indices: np.ndarray) -> dict:
    # indices expected to be 1D array
    return {int(idx): i for i, idx in enumerate(indices.tolist())}


def concat_by_index(body_npz: dict, fin_npz: dict) -> tuple[np.ndarray, np.ndarray]:
    body_idx = body_npz["original_index"]
    fin_idx = fin_npz["original_index"]

    body_feat = body_npz["embed_features1"]
    fin_feat = fin_npz["embed_features1"]

    body_map = build_index_map(body_idx)
    fin_map = build_index_map(fin_idx)

    # Keep order following body indices
    out_feat = []
    out_idx = []

    missing = 0
    for idx in body_idx.tolist():
        idx_int = int(idx)
        if idx_int not in fin_map:
            missing += 1
            continue
        out_idx.append(idx_int)
        out_feat.append(np.concatenate([body_feat[body_map[idx_int]], fin_feat[fin_map[idx_int]]], axis=0))

    out_idx = np.asarray(out_idx, dtype=np.int64)
    out_feat = np.asarray(out_feat)

    if missing > 0:
        print(f"[WARN] Missing {missing} indices in fin embeddings. Output size={len(out_idx)}")

    return out_idx, out_feat


def main():
    parser = argparse.ArgumentParser(description="Concat body+fin embeddings by original_index")
    parser.add_argument("--body", type=str, required=True)
    parser.add_argument("--fin", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    body_npz = load_npz(args.body)
    fin_npz = load_npz(args.fin)

    out_idx, out_feat = concat_by_index(body_npz, fin_npz)

    if args.normalize:
        out_feat = l2_normalize(out_feat)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(str(out_path), original_index=out_idx, embed_features1=out_feat)
    print(f"Saved: {out_path} | shape={out_feat.shape}")


if __name__ == "__main__":
    main()
