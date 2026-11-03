"""Small CPU fits exercise the real Lightning lifecycle and checkpoint format."""

import hashlib
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, default_collate

from phenostate import infer
from phenostate.data.module import PhenoStateDataModule
from phenostate.models import MAEConfig, model_from_checkpoint
from phenostate.train import mae as train_entry
from phenostate.train.module import PhenoStateModule


class TinyWells(Dataset):
    def __init__(self, count=3):
        self.images = torch.rand(count, 2, 16, 16, generator=torch.Generator().manual_seed(7))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"well_id": f"well-{index}", "image": self.images[index]}


def config(kind="mae"):
    return MAEConfig(model_type=kind, image_size=16, patch_size=8, in_channels=2,
                     encoder_dim=16, encoder_depth=1, encoder_heads=2,
                     decoder_dim=8, decoder_depth=1, decoder_heads=2, diffusion_steps=10).to_dict()


def trainer(path, **kwargs):
    return pl.Trainer(default_root_dir=path, accelerator="cpu", devices=1, precision="32-true",
                      logger=False, enable_model_summary=False, enable_progress_bar=False,
                      num_sanity_val_steps=0, **kwargs)


@pytest.mark.parametrize("kind", ["mae", "ca_mae", "diffmae", "cnn_mae"])
def test_lightning_fit_checkpoint_and_embedding(tmp_path, monkeypatch, kind):
    pl.seed_everything(17)
    wells = TinyWells()
    loader = DataLoader(wells, batch_size=1)
    root = tmp_path / "mds"
    root.mkdir()
    manifest = json.dumps({"format": "rxrx3-core-well-mds-v1", "complete": True,
                           "counts": {"train": 3, "val": 3}}).encode()
    (root / "manifest.json").write_bytes(manifest)
    module = PhenoStateModule(config(kind), total_steps=2,
                             manifest_sha256=hashlib.sha256(manifest).hexdigest())
    callback = ModelCheckpoint(dirpath=tmp_path / "checkpoints", filename="best", monitor="val_loss",
                               save_last=True)
    fit = trainer(tmp_path, max_epochs=1, accumulate_grad_batches=2, gradient_clip_val=1.0,
                  callbacks=[callback])
    fit.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert fit.global_step == 2
    checkpoint = Path(callback.best_model_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["phenostate_precision"] == "32-true"
    restored = model_from_checkpoint(state).eval()
    lightning_restored = PhenoStateModule.load_from_checkpoint(checkpoint).eval()
    module.eval()
    with torch.inference_mode():
        expected = module.model.encode(wells.images).numpy()
        np.testing.assert_allclose(restored.encode(wells.images).numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(lightning_restored.model.encode(wells.images).numpy(), expected, atol=1e-6)
    first = fit.validate(module, dataloaders=loader, verbose=False)[0]["val_loss"]
    second = fit.validate(module, dataloaders=loader, verbose=False)[0]["val_loss"]
    assert first == second

    @contextmanager
    def batches(*args, **kwargs):
        yield iter([{"well_id": [f"well-{i}" for i in range(3)], "image": wells.images}])

    monkeypatch.setattr(infer, "local_well_batches", batches)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    destination = tmp_path / "embedding.parquet"
    assert infer.export_embeddings(checkpoint, root, "train", destination) == 3
    table = pq.read_table(destination)
    np.testing.assert_allclose(table["embedding"].to_pylist(), expected, rtol=1e-5, atol=1e-6)
    assert table.schema.metadata[b"phenostate_model_type"].decode() == kind


def test_datamodule_manifest_and_hparams(tmp_path):
    data = {"format": "rxrx3-core-well-mds-v1", "complete": True, "counts": {"train": 3, "val": 2}}
    content = json.dumps(data).encode()
    (tmp_path / "manifest.json").write_bytes(content)
    module = PhenoStateDataModule(tmp_path)
    module.setup("fit")
    assert module.manifest_sha256 == hashlib.sha256(content).hexdigest()
    assert module.hparams.num_workers == 0
    data["complete"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete"):
        module.setup("fit")


def test_custom_checkpoints_are_rejected():
    with pytest.raises(ValueError, match="Lightning"):
        model_from_checkpoint({"model_config": config(), "model": {}})


def test_training_cli_native_outputs(tmp_path, monkeypatch):
    root, output = tmp_path / "mds", tmp_path / "run"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "format": "rxrx3-core-well-mds-v1", "complete": True, "counts": {"train": 3, "val": 3},
    }))
    configuration = tmp_path / "model.json"
    configuration.write_text(json.dumps(config()))
    data = PhenoStateDataModule(root)
    monkeypatch.setattr(data, "_loader", lambda split: DataLoader(TinyWells(), batch_size=1))
    monkeypatch.setattr(train_entry, "PhenoStateDataModule", lambda *args: data)
    monkeypatch.setattr(sys, "argv", ["phenostate-train", "--mds-root", str(root),
                        "--output", str(output), "--model-config", str(configuration),
                        "--accelerator", "cpu", "--epochs", "1", "--accumulation-steps", "2",
                        "--checkpoint-every", "1", "--log-every", "1"])
    train_entry.main()
    state = torch.load(output / "checkpoints/last.ckpt", weights_only=False)
    assert state["hyper_parameters"]["total_steps"] == state["global_step"] == 2
    assert state["datamodule_hyper_parameters"]["batch_size"] == 1
    assert (output / "checkpoints/best.ckpt").is_file()
    assert "val_loss" in (output / "logs/metrics.csv").read_text()


class StreamingBase(DataLoader):
    """Tiny stand-in for the public Streaming cursor, without cloud dependencies."""
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.epoch = self.next_epoch = self.cursor = self.offset = 0

    def __iter__(self):
        self.epoch = self.next_epoch
        self.next_epoch += 1
        start, self.offset = self.offset, 0
        self.cursor = start
        for index in range(start, len(self.dataset), self.batch_size):
            self.cursor = min(index + self.batch_size, len(self.dataset))
            yield default_collate([self.dataset[i] for i in range(index, self.cursor)])

    def state_dict(self):
        return {"epoch": self.epoch, "sample_in_epoch": self.cursor}

    def load_state_dict(self, state):
        self.next_epoch = state["epoch"]
        self.offset = self.cursor = state["sample_in_epoch"]


@pytest.fixture
def stateful_loader(monkeypatch):
    fake = types.ModuleType("streaming")
    fake.StreamingDataLoader, fake.StreamingDataset = StreamingBase, object
    monkeypatch.setitem(sys.modules, "streaming", fake)
    path = Path(__file__).parents[1] / "src/phenostate/data/streaming_dataset.py"
    spec = importlib.util.spec_from_file_location("phenostate.data._loader_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda count=3, batch_size=1: module.WellDataLoader(TinyWells(count), batch_size=batch_size)


class RecordingModule(PhenoStateModule):
    def __init__(self):
        super().__init__(config(), total_steps=6)
        self.seen = []

    def training_step(self, batch, batch_idx):
        self.seen.extend((self.current_epoch, key) for key in batch["well_id"])
        return super().training_step(batch, batch_idx)


@pytest.mark.parametrize("count,batch_size,accumulation,save_step", [(3, 1, 1, 2), (3, 1, 1, 3),
                                                                  (7, 2, 2, 1), (7, 2, 2, 2)])
def test_native_stateful_resume_at_mid_and_last_batch(
    tmp_path, stateful_loader, count, batch_size, accumulation, save_step,
):
    pl.seed_everything(19)
    callback = ModelCheckpoint(dirpath=tmp_path / "checkpoints", filename="step-{step}",
                               auto_insert_metric_name=False, every_n_train_steps=save_step, save_top_k=-1)
    first = trainer(tmp_path / "first", max_epochs=1, callbacks=[callback],
                    accumulate_grad_batches=accumulation)
    first.fit(RecordingModule(), train_dataloaders=stateful_loader(count, batch_size),
              val_dataloaders=DataLoader(TinyWells()))
    resumed = RecordingModule()
    second = trainer(tmp_path / "second", max_epochs=2, enable_checkpointing=False,
                     accumulate_grad_batches=accumulation)
    second.fit(resumed, train_dataloaders=stateful_loader(count, batch_size), val_dataloaders=DataLoader(TinyWells()),
               ckpt_path=tmp_path / "checkpoints" / f"step-{save_step}.ckpt")
    offset = min(save_step * batch_size * accumulation, count)
    expected = [(0, f"well-{i}") for i in range(offset, count)] + [(1, f"well-{i}") for i in range(count)]
    assert resumed.seen == expected
    pl.seed_everything(19)
    baseline = RecordingModule()
    full = trainer(tmp_path / "full", max_epochs=2, enable_checkpointing=False,
                   accumulate_grad_batches=accumulation)
    full.fit(baseline, train_dataloaders=stateful_loader(count, batch_size),
             val_dataloaders=DataLoader(TinyWells()))
    assert second.global_step == full.global_step
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
