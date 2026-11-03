"""Lightning DataModule for the fixed, complete MDS plate split."""

import hashlib
import json
from pathlib import Path

import lightning.pytorch as pl


class PhenoStateDataModule(pl.LightningDataModule):
    def __init__(self, mds_root, batch_size=1, num_workers=0, seed=2026):
        super().__init__()
        if batch_size < 1 or num_workers < 0:
            raise ValueError("Invalid batch size or worker count")
        self.save_hyperparameters({"mds_root": str(Path(mds_root).resolve()),
                                   "batch_size": batch_size, "num_workers": num_workers, "seed": seed})
        self.manifest = None
        self.manifest_sha256 = None
        self._loaders = {}

    def setup(self, stage=None):
        content = (Path(self.hparams.mds_root) / "manifest.json").read_bytes()
        manifest = json.loads(content)
        if manifest.get("format") != "rxrx3-core-well-mds-v1" or not manifest.get("complete"):
            raise ValueError("Training requires a complete RxRx3-core MDS conversion")
        if any(manifest["counts"][split] < 1 for split in ("train", "val")):
            raise ValueError("Training and validation splits must both be populated")
        self.manifest = manifest
        self.manifest_sha256 = hashlib.sha256(content).hexdigest()

    def _loader(self, split):
        from .streaming_dataset import make_dataloader

        if split not in self._loaders:
            self._loaders[split] = make_dataloader(
                mds_root=self.hparams.mds_root, split=split, batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers, shuffle_seed=self.hparams.seed,
            )
        return self._loaders[split]

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")
