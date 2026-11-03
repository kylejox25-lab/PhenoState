"""One Lightning training module for the four image autoencoders."""

import math

import lightning.pytorch as pl
import torch

from phenostate.models import MAEConfig, build_model


def learning_rate(step, total_steps, peak, warmup_fraction):
    warmup = int(total_steps * warmup_fraction)
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


class PhenoStateModule(pl.LightningModule):
    def __init__(
        self, model_config: dict, total_steps: int, learning_rate: float = 1.5e-4,
        weight_decay: float = 0.05, warmup_fraction: float = 0.05,
        seed: int = 2026, manifest_sha256: str | None = None,
    ):
        super().__init__()
        if total_steps < 1 or learning_rate <= 0 or weight_decay < 0 or not 0 <= warmup_fraction < 1:
            raise ValueError("Invalid optimizer or warmup settings")
        self.save_hyperparameters()
        self.model = build_model(MAEConfig(**model_config))

    def forward(self, image, *, generator=None):
        return self.model(image, generator=generator)

    def _step(self, batch, batch_idx, stage):
        image = batch["image"]
        offset = (self.current_epoch * int(self.trainer.num_training_batches)
                  if stage == "train" else 10_000)
        generator = torch.Generator(device=image.device).manual_seed(self.hparams.seed + offset + batch_idx)
        loss = self(image, generator=generator)["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss, wells={batch.get('well_id')}")
        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, batch_size=len(image))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        return {"well_id": batch["well_id"], "embedding": self.model.encode(batch["image"])}

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            target = no_decay if parameter.ndim == 1 or name in {
                "encoder_pos", "decoder_pos", "mask_token",
            } else decay
            target.append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate(step, self.hparams.total_steps, 1.0,
                                                  self.hparams.warmup_fraction),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["phenostate_precision"] = str(self.trainer.precision)

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint["hyper_parameters"]
        if saved != dict(self.hparams):
            raise ValueError("Resume requires the same model, optimizer, schedule, seed and MDS manifest")
