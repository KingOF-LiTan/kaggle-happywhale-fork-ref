from logging import getLogger
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader

from run.init.dataset import init_datasets_from_config
from run.init.forwarder import Forwarder
from run.init.model import init_model_from_config
from run.init.optimizer import init_optimizer_from_config
from run.init.preprocessing import Preprocessing
from run.init.scheduler import init_scheduler_from_config
from src.datasets.wrapper import WrapperDataset

logger = getLogger(__name__)


def calc_map5(topk, label):
    topk = topk.tolist()
    label = label.tolist()
    res = []
    for i in range(len(topk)):
        if label[i] not in topk[i]:
            res.append(0)
        else:
            res.append(1 / (topk[i].index(label[i]) + 1))
    return np.array(res).mean()


def calc_acc(topk, label):
    topk = topk.tolist()
    label = label.tolist()
    res = []
    for i in range(len(topk)):
        if label[i] == topk[i][0]:
            res.append(1)
        else:
            res.append(0)
    return np.array(res).mean()


class PLModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg.copy()

        pretrained = False if cfg.training.debug else True
        model = init_model_from_config(cfg.model, pretrained=pretrained)
        self.forwarder = Forwarder(cfg.forwarder, model)

        raw_datasets = init_datasets_from_config(cfg.dataset)

        preprocessing = Preprocessing(cfg.augmentation, **cfg.preprocessing)
        self.datasets = {}
        transforms = {
            "train": preprocessing.get_train_transform(),
            "val": preprocessing.get_val_transform(),
            "test": preprocessing.get_test_transform(),
        }
        for phase in ["train", "val", "test"]:
            self.datasets[phase] = WrapperDataset(
                raw_datasets[phase], transforms[phase], phase
            )
            logger.info(f"{phase}: {len(self.datasets[phase])}")

    def on_train_epoch_start(self):
        if self.cfg.get("freeze_bn_stats", False):
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        additional_info = {}
        _, loss, _, _, _ = self.forwarder.forward(
            batch, phase="train", epoch=self.current_epoch, **additional_info
        )

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["image"].shape[0],
        )
        return loss

    def _end_process(self, outputs: List[Dict[str, Tensor]], phase: str):
        # Aggregate results
        epoch_results: Dict[str, np.ndarray] = {}
        outputs = self.all_gather(outputs)

        for key in [
            "original_index",
            "file_name",
            "label",
            "pred",
            "embed_features1",
            "embed_features2",
            "label_species",
            "pred_species",
        ]:
            if isinstance(outputs[0][key], Tensor):
                result = torch.cat([torch.atleast_1d(x[key]) for x in outputs], dim=0)
                result = torch.flatten(result, start_dim=0, end_dim=0)
                epoch_results[key] = result.detach().cpu().numpy()
            else:
                result = np.concatenate([x[key] for x in outputs])
                epoch_results[key] = result

        pred = epoch_results["pred"]
        if phase == "test" and self.trainer.global_rank == 0:
            # Save test results ".npz" format
            epoch_results.pop("pred")
            epoch_results["pred_logit"] = -np.sort(-pred)[:, :1000]
            epoch_results["pred_idx"] = np.argsort(-pred)[:, :1000]
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)
            np.savez_compressed(
                str(test_results_filepath / "test_results.npz"),
                **epoch_results,
            )

        loss = (
            torch.cat([torch.atleast_1d(x["loss"]) for x in outputs])
            .detach()
            .cpu()
            .numpy()
        )
        mean_loss = np.mean(loss)

        label = epoch_results["label"]
        topk = np.argsort(-pred)[:, :5]
        map5 = calc_map5(topk, label)
        acc = calc_acc(topk, label)
        label_species = epoch_results["label_species"]
        if len(epoch_results["pred_species"].shape) != 1:
            pred_species = np.argsort(-epoch_results["pred_species"])[:, :1]
            acc_species = calc_acc(pred_species, label_species)
            self.log(f"{phase}/acc_species", acc_species, prog_bar=True)
        # Log items
        self.log(f"{phase}/loss", mean_loss, prog_bar=True)
        self.log(f"{phase}/map5", map5, prog_bar=True)
        self.log(f"{phase}/acc", acc, prog_bar=True)

    def _evaluation_step(self, batch: Dict[str, Tensor], phase: Literal["val", "test"]):
        (
            preds,
            loss,
            embed_features1,
            preds_species,
            embed_features2,
        ) = self.forwarder.forward(batch, phase=phase, epoch=self.current_epoch)
        if preds_species is not None:
            preds_species = preds_species.detach()
        else:
            preds_species = batch["label_species"]
        output = {
            "loss": loss,
            "label": batch["label"],
            "label_species": batch["label_species"],
            "original_index": batch["original_index"],
            "file_name": batch["file_name"],
            "pred": preds.detach(),
            "pred_species": preds_species,
            "embed_features1": embed_features1.detach(),
            "embed_features2": embed_features2.detach(),
        }
        return output

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        out = self._evaluation_step(batch, phase="val")
        if not hasattr(self, "_val_outputs") or self._val_outputs is None:
            self._val_outputs = []
        self._val_outputs.append(out)
        return out

    def on_validation_epoch_end(self) -> None:
        outputs = getattr(self, "_val_outputs", None)
        if outputs is None:
            return
        self._end_process(outputs, "val")
        self._val_outputs = []

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        out = self._evaluation_step(batch, phase="test")
        if not hasattr(self, "_test_outputs") or self._test_outputs is None:
            self._test_outputs = []
        self._test_outputs.append(out)
        return out

    def on_test_epoch_end(self) -> None:
        outputs = getattr(self, "_test_outputs", None)
        if outputs is None:
            return
        self._end_process(outputs, "test")
        self._test_outputs = []

    def configure_optimizers(self):
        model = self.forwarder.model
        opt_cls, kwargs = init_optimizer_from_config(
            self.cfg.optimizer, model.forward_features.parameters()
        )

        self.cfg.optimizer.lr = self.cfg.optimizer.lr_head
        kwargs_head = init_optimizer_from_config(
            self.cfg.optimizer, model.head.parameters(), return_cls=False
        )

        self.cfg.optimizer.lr = self.cfg.optimizer.lr_head_species
        kwargs_head_species = init_optimizer_from_config(
            self.cfg.optimizer, model.head_species.parameters(), return_cls=False
        )

        optimizer = opt_cls([kwargs, kwargs_head, kwargs_head_species])
        scheduler = init_scheduler_from_config(self.cfg.scheduler, optimizer)

        if scheduler is None:
            return [optimizer]

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _dataloader(self, phase: str) -> DataLoader:
        logger.info(f"{phase} data loader called")
        dataset = self.datasets[phase]

        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.training.num_workers

        num_gpus = self.cfg.training.num_gpus
        if phase != "train":
            batch_size = self.cfg.training.batch_size_test
        batch_size //= num_gpus
        num_workers //= num_gpus

        drop_last = True if self.cfg.training.drop_last and phase == "train" else False
        shuffle = phase == "train"

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers = (num_workers > 0),
            pin_memory = True,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(phase="train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(phase="val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(phase="test")
