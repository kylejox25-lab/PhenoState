"""Export completion and provenance checks, independent of image storage backend."""

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from phenostate import infer
from phenostate.models import MAEConfig, build_model


@pytest.fixture
def export_case(tmp_path, monkeypatch):
    root = tmp_path / "mds"
    root.mkdir()
    manifest_bytes = json.dumps({
        "format": "rxrx3-core-well-mds-v1", "complete": True,
        "counts": {"train": 3, "val": 1, "test": 1},
    }).encode()
    (root / "manifest.json").write_bytes(manifest_bytes)
    checkpoint = tmp_path / "weights.pt"
    torch.save({
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "run_config": {"amp": False},
    }, checkpoint)

    class Model:
        config = SimpleNamespace(encoder_dim=2, model_type="mae")

        def to(self, device):
            return self

        def eval(self):
            return self

        def encode(self, image):
            return image

    images = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    data = {"well_id": ["a", "b", "c"], "image": images}

    @contextmanager
    def batches(*args, **kwargs):
        yield iter([data])

    monkeypatch.setattr(infer, "local_well_batches", batches)
    monkeypatch.setattr(infer, "model_from_checkpoint", lambda state: Model())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return SimpleNamespace(root=root, checkpoint=checkpoint, output=tmp_path / "features.parquet",
                           data=data, images=images)


def run_export(case, **kwargs):
    return infer.export_embeddings(case.checkpoint, case.root, "train", case.output, **kwargs)


def test_export_commits_only_complete_finite_unique_wells(export_case):
    case = export_case
    assert run_export(case) == 3
    table = pq.read_table(case.output)
    assert table["well_id"].to_pylist() == ["a", "b", "c"]
    np.testing.assert_array_equal(table["embedding"].to_pylist(), case.images.numpy())
    assert not case.output.with_suffix(".parquet.incomplete").exists()
    tags = table.schema.metadata
    assert tags[b"cellvit_split"] == b"train"
    assert tags[b"cellvit_loader"] == b"local-threaded"
    assert tags[b"cellvit_checkpoint_sha256"].decode() == hashlib.sha256(case.checkpoint.read_bytes()).hexdigest()


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite", "empty", "nonstring"])
def test_bad_batches_never_publish_output(export_case, problem):
    case = export_case
    if problem == "missing":
        case.data["well_id"] = ["a", "b"]
        case.data["image"] = case.images[:2]
    elif problem == "duplicate":
        case.data["well_id"] = ["a", "a", "c"]
    elif problem == "empty":
        case.data["well_id"] = ["a", "", "c"]
    elif problem == "nonstring":
        case.data["well_id"] = ["a", None, "c"]
    else:
        case.data["image"][0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        run_export(case)
    assert not case.output.exists()
    assert case.output.with_suffix(".parquet.incomplete").exists()


def test_incomplete_file_is_not_reused_even_with_skip_existing(export_case):
    case = export_case
    case.output.with_suffix(".parquet.incomplete").write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        run_export(case, skip_existing=True)


def test_existing_output_requires_explicit_skip(export_case):
    case = export_case
    run_export(case)
    with pytest.raises(FileExistsError):
        run_export(case)


def test_verified_skip_does_not_load_model_or_checkpoint(export_case, monkeypatch):
    case = export_case
    run_export(case)
    original = case.output.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("verified existing file must not load model weights")

    monkeypatch.setattr(infer, "model_from_checkpoint", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    assert run_export(case, skip_existing=True) == 3
    assert case.output.read_bytes() == original


def test_skip_rejects_changed_checkpoint(export_case):
    case = export_case
    run_export(case)
    case.checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError):
        run_export(case, skip_existing=True)


@pytest.mark.parametrize("valid", [True, False])
def test_skip_closes_reader_on_success_and_error(export_case, monkeypatch, valid):
    case = export_case
    run_export(case)
    if not valid:
        case.checkpoint.write_bytes(b"different weights")
    closed = []
    original_close = pq.ParquetFile.close

    def close(reader, *args, **kwargs):
        closed.append(reader)
        return original_close(reader, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "close", close)
    if valid:
        assert run_export(case, skip_existing=True) == 3
    else:
        with pytest.raises(ValueError):
            run_export(case, skip_existing=True)
    assert closed


@pytest.mark.parametrize("problem", ["duplicate", "empty", "nonfinite", "missing", "legacy"])
def test_skip_rejects_invalid_existing_content(export_case, problem):
    case = export_case
    run_export(case)
    table = pq.read_table(case.output)
    ids = ["a", "b", "c"]
    values = case.images.numpy().copy()
    metadata = dict(table.schema.metadata)
    if problem == "duplicate":
        ids[1] = ids[0]
    elif problem == "empty":
        ids[1] = ""
    elif problem == "nonfinite":
        values[0, 0] = float("inf")
    elif problem == "missing":
        ids = ids[:2]
        values = values[:2]
    else:
        metadata.pop(b"cellvit_checkpoint_sha256")
    replacement = pa.Table.from_arrays([
        pa.array(ids, type=pa.string()),
        pa.array(values.tolist(), type=table.schema.field("embedding").type),
    ], schema=table.schema.with_metadata(metadata))
    pq.write_table(replacement, case.output)
    with pytest.raises(ValueError):
        run_export(case, skip_existing=True)


@pytest.mark.parametrize("kind", ["mae", "ca_mae", "diffmae", "cnn_mae"])
def test_export_matches_frozen_encoder_for_all_architectures(tmp_path, monkeypatch, kind):
    config = MAEConfig(
        model_type=kind, image_size=16, patch_size=8, in_channels=2,
        encoder_dim=16, encoder_depth=1, encoder_heads=2,
        decoder_dim=8, decoder_depth=1, decoder_heads=2, diffusion_steps=10,
    )
    model = build_model(config).eval()
    images = torch.rand(3, 2, 16, 16)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        expected = model.encode(images).numpy()
    root = tmp_path / "mds"
    root.mkdir()
    manifest = json.dumps({"format": "rxrx3-core-well-mds-v1", "complete": True,
                           "counts": {"train": 3}}).encode()
    (root / "manifest.json").write_bytes(manifest)
    checkpoint = tmp_path / "weights.pt"
    torch.save({"format": "cellvit-masked-autoencoder-v2", "model_config": config.to_dict(),
                "model": before, "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "run_config": {"amp": False}}, checkpoint)

    @contextmanager
    def batches(*args, **kwargs):
        yield iter([{"well_id": ["a", "b", "c"], "image": images}])

    monkeypatch.setattr(infer, "local_well_batches", batches)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "features.parquet"
    assert infer.export_embeddings(checkpoint, root, "train", output) == 3
    actual = np.asarray(pq.read_table(output)["embedding"].to_pylist())
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert all(torch.equal(value, restored["model"][key]) for key, value in before.items())
