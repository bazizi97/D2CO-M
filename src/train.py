"""Unified training and evaluation handler for DIFUSCO CO models (CFN & MAX2SAT).

Usage examples
--------------
# Train CFN model:
    python train.py task=cfn

# Train MAX2SAT model:
    python train.py task=maxsat

# Override parameters:
    python train.py task=maxsat model.learning_rate=1e-4 data.batch_size=32

# Test only (no training):
    python train.py do_train=false do_test=true ckpt_path=/path/to/ckpt.ckpt
"""

import argparse
import datetime
import os
import shutil

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from cfn_model_sparse import CFNSPARSEModel
from pl_maxsat_model import MAX2SATModel


def _cfg_to_namespace(cfg: DictConfig):
    """Flatten the composed Hydra DictConfig into a simple namespace object."""
    flat = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    ns = argparse.Namespace()
    for key, value in flat.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(ns, sub_key, sub_value)
        else:
            setattr(ns, key, value)
    return ns


@hydra.main(config_path="../conf_cpd", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    rank_zero_info("=== Effective config ===\n" + OmegaConf.to_yaml(cfg))

    args = _cfg_to_namespace(cfg)
    task = getattr(args, "task", "cfn").lower()

    # -----------------------------
    # Task & Model Selection
    # -----------------------------
    if task == "cfn":
        model_class = CFNSPARSEModel
        monitor_metric = getattr(args, "monitor_metric", "val/NSR")
        early_stop_metric = getattr(args, "early_stop_metric", "val/NSR")
        saving_mode = getattr(args, "saving_mode", "max")
    elif task == "maxsat":
        model_class = MAX2SATModel
        monitor_metric = getattr(args, "monitor_metric", "val/satisfied_clauses")
        early_stop_metric = getattr(args, "early_stop_metric", "val/satisfied_clauses")
        saving_mode = getattr(args, "saving_mode", "max")
    else:
        raise NotImplementedError(f"Unknown task: {task}")

    rank_zero_info(f"Task selected: {task} | Model: {model_class.__name__}")

    model = model_class(param_args=args)

    # -----------------------------
    # Directory Setup & Config Snapshot
    # -----------------------------
    models_dir = os.path.join(args.storage_path, "models")
    configs_dir = os.path.join(args.storage_path, "configs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    run_label = getattr(args, "wandb_logger_name", None) or f"{task}_run"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = os.path.join(configs_dir, f"{run_label}_{timestamp}.yaml")
    OmegaConf.save(cfg, config_save_path)
    rank_zero_info(f"[Config] Saved configuration -> {config_save_path}")

    # -----------------------------
    # Logger Setup
    # -----------------------------
    use_wandb = getattr(args, "use_wandb", True)
    if use_wandb:
        wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
        logger = WandbLogger(
            name=run_label,
            project=getattr(args, "project_name", "D2CO-M"),
            entity=getattr(args, "wandb_entity", None),
            save_dir=models_dir,
            id=getattr(args, "resume_id", None) or wandb_id,
        )
        logger_save_dir = logger.save_dir
        logger_name = logger.name
        logger_id = logger._id
        ckpt_dir = os.path.join(models_dir, run_label, logger_id, "checkpoints")
        rank_zero_info(f"[WandbLogger] Logging to {logger_save_dir}/{logger_name}/{logger_id}")
    else:
        logger = CSVLogger(save_dir=models_dir, name=run_label)
        logger_save_dir = logger.save_dir
        logger_name = logger.name
        logger_id = f"version_{logger.version if logger.version is not None else 0}"
        ckpt_dir = os.path.join(models_dir, run_label, logger_id, "checkpoints")
        rank_zero_info(f"[CSVLogger] Logging locally to {ckpt_dir}")

    # -----------------------------
    # Callbacks Setup
    # -----------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode=saving_mode,
        save_top_k=getattr(args, "save_top_k", 3),
        save_last=True,
        dirpath=ckpt_dir,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    early_stop_patience = getattr(args, "early_stop_patience", 10)
    early_stop_callback = EarlyStopping(
        monitor=early_stop_metric,
        mode=saving_mode,
        patience=early_stop_patience,
        min_delta=getattr(args, "early_stop_delta", 1e-4),
        verbose=True,
        check_on_train_epoch_end=False,
    )

    # -----------------------------
    # Trainer Setup
    # -----------------------------
    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=args.num_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            checkpoint_callback,
            lr_callback,
            early_stop_callback,
        ],
        logger=logger,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(static_graph=True),
        precision=16 if getattr(args, "fp16", False) else 32,
        log_every_n_steps=20,
    )

    rank_zero_info(f"{'-' * 100}\n{str(model.model)}\n{'-' * 100}\n")

    # -----------------------------
    # Execution (Fit / Validate / Test)
    # -----------------------------
    ckpt_path = getattr(args, "ckpt_path", None)

    if getattr(args, "do_train", True):
        if getattr(args, "resume_weight_only", False):
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if getattr(args, "do_test", False):
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    elif getattr(args, "do_test", False):
        if getattr(args, "do_valid_only", False):
            trainer.validate(model, ckpt_path=ckpt_path)
        else:
            trainer.test(model, ckpt_path=ckpt_path)

    if use_wandb:
        trainer.logger.finalize("success")


if __name__ == "__main__":
    main()
