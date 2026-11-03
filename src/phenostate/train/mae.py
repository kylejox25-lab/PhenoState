"""Train a PhenoState autoencoder with PyTorch Lightning."""

import argparse
import json
import math
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from phenostate.data.module import PhenoStateDataModule
from phenostate.models import MAEConfig
from phenostate.train.module import PhenoStateModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mds-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader processes; default 0 avoids multiprocessing temp files on NFS")
    parser.add_argument("--accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--precision", choices=("16-mixed", "32-true"))
    parser.add_argument("--resume", type=Path, help="Lightning .ckpt; use the same model and data settings")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    if min(args.epochs, args.accumulation_steps, args.checkpoint_every, args.log_every) < 1:
        parser.error("Epochs, accumulation and logging/checkpoint intervals must be positive")
    if args.max_steps != -1 and args.max_steps < 1:
        parser.error("max-steps must be positive or -1")
    if args.accelerator == "cpu" and args.precision == "16-mixed":
        parser.error("Use --precision 32-true on CPU")
    root, output = args.mds_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Training output must be outside the MDS directory")
    if args.resume is None and output.exists():
        raise FileExistsError(f"Run directory already exists: {output}")
    if args.resume is not None and (not args.resume.is_file() or args.resume.suffix != ".ckpt"):
        parser.error("resume must point to a Lightning .ckpt file")

    pl.seed_everything(args.seed, workers=True)
    data = PhenoStateDataModule(root, args.batch_size, args.num_workers, args.seed)
    data.setup("fit")
    total_steps = args.epochs * math.ceil(
        math.ceil(data.manifest["counts"]["train"] / args.batch_size) / args.accumulation_steps,
    )
    if args.max_steps != -1:
        total_steps = min(total_steps, args.max_steps)
    config = MAEConfig(**json.loads(args.model_config.read_text(encoding="utf-8"))) if args.model_config else MAEConfig()
    module = PhenoStateModule(
        config.to_dict(), total_steps=total_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction, seed=args.seed, manifest_sha256=data.manifest_sha256,
    )
    checkpoint_dir = output / "checkpoints"
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, filename="best", monitor="val_loss", mode="min",
                        save_top_k=1, save_last=True, enable_version_counter=False),
        ModelCheckpoint(dirpath=checkpoint_dir, filename="step-{step:08d}",
                        every_n_train_steps=args.checkpoint_every, save_top_k=1,
                        auto_insert_metric_name=False, enable_version_counter=False),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        default_root_dir=output, accelerator=args.accelerator, devices=1,
        precision=args.precision or ("16-mixed" if args.accelerator == "gpu" else "32-true"),
        max_epochs=args.epochs, max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulation_steps, gradient_clip_val=1.0,
        log_every_n_steps=args.log_every, num_sanity_val_steps=0,
        callbacks=callbacks, logger=CSVLogger(str(output), name="logs", version=""),
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule=data, ckpt_path=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
