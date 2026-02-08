import logging
import os
from pathlib import Path

import hydra
import numpy as np

if not hasattr(np, "int"):
    np.int = int
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from run.pl_model import PLModel

logger = logging.getLogger(__name__)


def prepare_env() -> None:
    logging.getLogger("PIL").setLevel(logging.WARNING)
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)
    os.environ.setdefault("PYTHONPATH", ".")


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.inference_mode()
def _extract(
    cfg: DictConfig,
    model: PLModel,
    phase: str,
    ckpt_path: "str | None",
    out_path: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path:
        logger.info(f"Loading ckpt state_dict from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    if phase == "train":
        ds = model.datasets["train"]
        batch_size = cfg.training.batch_size
        shuffle = False
        drop_last = False
    elif phase == "val":
        ds = model.datasets["val"]
        batch_size = cfg.training.batch_size_test
        shuffle = False
        drop_last = False
    elif phase == "test":
        ds = model.datasets["test"]
        batch_size = cfg.training.batch_size_test
        shuffle = False
        drop_last = False
    else:
        raise ValueError(f"Unknown phase: {phase}")

    num_workers = int(cfg.training.num_workers)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    original_index = []
    embed1 = []
    embed2 = []

    pbar = tqdm(loader, desc=f"extract_{phase}")
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        logits, loss, e1, logits_species, e2 = model.forwarder.forward(
            batch, phase="test", epoch=0
        )
        original_index.append(_as_numpy(batch["original_index"]))
        embed1.append(_as_numpy(e1))
        if e2 is not None:
            embed2.append(_as_numpy(e2))

    res = {
        "original_index": np.concatenate(original_index, axis=0),
        "embed_features1": np.concatenate(embed1, axis=0),
    }
    if len(embed2) > 0:
        res["embed_features2"] = np.concatenate(embed2, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **res)
    logger.info(f"Saved embeddings: {out_path}")


@hydra.main(config_path="conf", config_name="config")
def entry(cfg: DictConfig) -> None:
    prepare_env()

    # NOTE: reuse PLModel init to guarantee dataset / preprocessing alignment
    model = PLModel(cfg)

    # Use flat config keys to avoid hydra override-grammar issues on older versions
    ckpt_path = cfg.get("extract_ckpt_path", None)
    phase = cfg.get("extract_phase", "test")
    out = cfg.get("extract_out", None)
    if out is None:
        raise ValueError("Please set extract_out")

    _extract(cfg, model, phase=phase, ckpt_path=ckpt_path, out_path=Path(out))


if __name__ == "__main__":
    entry()
