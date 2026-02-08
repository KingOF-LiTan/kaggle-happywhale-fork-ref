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
    return {int(idx): i for i, idx in enumerate(indices.tolist())}

def concat_with_fallback(body_npz: dict, fin_npz: dict, fallback_strategy: str = "body") -> tuple[np.ndarray, np.ndarray]:
    body_idx = body_npz["original_index"]
    fin_idx = fin_npz["original_index"]
    body_feat = body_npz["embed_features1"]
    fin_feat = fin_npz["embed_features1"]

    body_map = build_index_map(body_idx)
    fin_map = build_index_map(fin_idx)

    out_feat = []
    out_idx = []
    
    missing_count = 0
    total_count = len(body_idx)
    
    fin_dim = fin_feat.shape[1]

    for idx in body_idx.tolist():
        idx_int = int(idx)
        b_idx = body_map[idx_int]
        b_vec = body_feat[b_idx]
        
        if idx_int in fin_map:
            f_idx = fin_map[idx_int]
            f_vec = fin_feat[f_idx]
        else:
            missing_count += 1
            if fallback_strategy == "body":
                # Use body features as a substitute for fin
                # Note: if dimensions differ, this might need padding or repeat
                if b_vec.shape[0] == fin_dim:
                    f_vec = b_vec
                else:
                    # If dims differ, use zeros
                    f_vec = np.zeros(fin_dim, dtype=b_vec.dtype)
            else:
                f_vec = np.zeros(fin_dim, dtype=b_vec.dtype)
        
        out_idx.append(idx_int)
        out_feat.append(np.concatenate([b_vec, f_vec], axis=0))

    print(f"Concat finished. Total: {total_count}, Fin Missing (Fallback used): {missing_count} ({missing_count/total_count:.2%})")
    
    return np.array(out_idx, dtype=np.int64), np.array(out_feat)

def main():
    parser = argparse.ArgumentParser(description="Concat body+fin embeddings with fallback strategy")
    parser.add_argument("--body", type=str, required=True, help="Path to body embeddings .npz")
    parser.add_argument("--fin", type=str, required=True, help="Path to fin embeddings .npz")
    parser.add_argument("--out", type=str, required=True, help="Output path .npz")
    parser.add_argument("--strategy", type=str, default="body", choices=["body", "zero"], help="Fallback strategy for missing fin")
    parser.add_argument("--normalize", action="store_true", help="Perform L2 normalization after concat")

    args = parser.parse_args()

    body_npz = load_npz(args.body)
    fin_npz = load_npz(args.fin)

    out_idx, out_feat = concat_with_fallback(body_npz, fin_npz, args.strategy)

    if args.normalize:
        out_feat = l2_normalize(out_feat)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(str(out_path), original_index=out_idx, embed_features1=out_feat)
    print(f"Saved: {out_path} | shape={out_feat.shape}")

if __name__ == "__main__":
    main()
