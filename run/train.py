import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from run.pl_model import PLModel

if not hasattr(np, "int"):
    np.int = int

logger = logging.getLogger(__name__)


def main(cfg: DictConfig, pl_model: type) -> Path:
    seed_everything(cfg.training.seed)
    out_dir = Path(cfg.out_dir).resolve()

    if cfg.test_model is not None:
        # Only run test with the given model weights
        is_test_mode = True
    else:
        # Run full training
        is_test_mode = False

    # init experiment logger
    if not cfg.training.use_wandb or is_test_mode:
        pl_logger = False
    else:
        pl_logger = WandbLogger(
            project=cfg.training.project_name,
            save_dir=str(out_dir),
            name=Path(out_dir).name,
        )

    # init lightning model
    model = pl_model(cfg)

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.training.monitor,
        mode=cfg.training.monitor_mode,
        save_top_k=1,
        save_last=True,
    )

    # init trainer
    def _init_trainer(resume=True):
        resume_from = cfg.training.resume_from if resume else None
        num_gpus = int(cfg.training.num_gpus)
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        if num_gpus > 1:
            strategy = "ddp"
            sync_batchnorm = True
        else:
            strategy = "auto"
            sync_batchnorm = False

        return Trainer(
            # env
            default_root_dir=str(out_dir),
            accelerator=accelerator,
            devices=num_gpus if accelerator == "gpu" else None,
            strategy=strategy,
            precision="16-mixed" if cfg.training.use_amp and accelerator == "gpu" else 32,
            # training
            fast_dev_run=cfg.training.debug,  # run only 1 train batch and 1 val batch
            enable_model_summary=False if cfg.training.debug else True,
            max_epochs=cfg.training.epoch,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            callbacks=[checkpoint_cb],
            logger=pl_logger,
            num_sanity_val_steps=0 if is_test_mode else 2,
            sync_batchnorm=sync_batchnorm,
        )

    trainer = _init_trainer()

    initial_best_score = None
    initial_best_model = None

    if cfg.training.resume_from is not None:
        ckpt = torch.load(cfg.training.resume_from, map_location="cpu", weights_only=False)
        try:
            cb = ckpt.get("callbacks", {})
            mc_state = None
            # 兼容旧版：key 可能是类、字符串、或者其它标识
            if isinstance(cb, dict):
                mc_state = cb.get(ModelCheckpoint, None) or cb.get("ModelCheckpoint", None)
            if mc_state is not None:
                initial_best_score = mc_state["best_model_score"].detach().cpu().numpy()
                initial_best_model = mc_state["best_model_path"]
                logger.info(
                    f"Initial best model ({initial_best_score:.4f}): {initial_best_model}"
                )
            else:
                logger.warning(
                    "ModelCheckpoint state not found in ckpt['callbacks'], skip initial best score tracking."
                )
        except Exception as e:
            logger.warning(f"Failed to read initial best score from resume checkpoint: {e}")
        finally:
            del ckpt

    resume_from = cfg.training.resume_from

    if is_test_mode:
        trainer.test(model, ckpt_path=cfg.test_model)
    else:
        trainer.fit(model, ckpt_path=resume_from)

        if cfg.training.resume_from is None:
            trainer.test(model, ckpt_path="best")  # test with the best checkpoint
        else:
            current_best_score = (
                trainer.checkpoint_callback.best_model_score.detach().cpu().numpy()
            )
            current_larger_than_initial = current_best_score > initial_best_score
            mode = trainer.checkpoint_callback.mode
            best_updated = (mode == "max" and current_larger_than_initial) or (
                mode == "min" and not current_larger_than_initial
            )
            if best_updated:
                best_ckpt = trainer.checkpoint_callback.best_model_path
                logger.info("The best model is updated.")
            else:
                best_ckpt = initial_best_model
                logger.info("The best model isn't changed.")

            current_epoch = trainer.current_epoch
            try:
                state_dict = torch.load(best_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            except FileNotFoundError:
                time.sleep(30)
                state_dict = torch.load(best_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            trainer = _init_trainer(resume=False)
            trainer.current_epoch = current_epoch
            logger.info(f"Testing with the best ckpt: {best_ckpt}")
            trainer.test(model)

        # extract weights and save
        if trainer.global_rank == 0:
            weights_path = str(Path(checkpoint_cb.dirpath) / "model_weights.pth")
            logger.info(f"Extracting and saving weights: {weights_path}")
            torch.save(model.forwarder.model.state_dict(), weights_path)

    # return path to checkpoints directory
    if checkpoint_cb.dirpath is not None:
        return Path(checkpoint_cb.dirpath)


def prepare_env() -> None:
    # Disable PIL's debug logs
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # move to original directory
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    # set PYTHONPATH if not set for possible launching of DDP processes
    os.environ.setdefault("PYTHONPATH", ".")


@hydra.main(config_path="conf", config_name="config")
def entry(cfg: DictConfig) -> None:
    prepare_env()
    main(cfg, PLModel)


if __name__ == "__main__":
    entry()
