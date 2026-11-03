"""Local inference loading must avoid IPC and keep prefetch memory bounded."""

import sys
import tempfile
import threading
import types
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from phenostate.data import inference_loader as il


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    samples = [
        {"well_id": f"well-{index}", "image": np.full((2, 3, 3), index, dtype=np.float32)}
        for index in range(5)
    ]
    readers = []

    class LocalDataset:
        def __init__(self, *, local, split):
            assert str(local) == str(tmp_path)
            assert split == "train"
            self.shards = readers

        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    module = types.ModuleType("streaming")
    module.LocalDataset = LocalDataset
    monkeypatch.setitem(sys.modules, "streaming", module)
    monkeypatch.setattr(il, "decode_well", lambda sample: sample["image"])
    (tmp_path / "train").mkdir()
    return tmp_path, samples, readers


@pytest.mark.parametrize("workers", [0, 2])
def test_batches_keep_order_and_last_partial_batch(local_data, workers):
    root, _, _ = local_data
    with il.local_well_batches(root, "train", batch_size=2, num_workers=workers,
                              expected_wells=5) as batches:
        result = list(batches)
    assert [batch["well_id"] for batch in result] == [
        ["well-0", "well-1"], ["well-2", "well-3"], ["well-4"],
    ]
    assert [batch["image"].shape[0] for batch in result] == [2, 2, 1]
    np.testing.assert_array_equal(result[-1]["image"].numpy(), np.full((1, 2, 3, 3), 4))


def test_declared_count_mismatch_is_rejected(local_data):
    root, _, _ = local_data
    with pytest.raises(ValueError, match="manifest expects"):
        with il.local_well_batches(root, "train", expected_wells=6):
            pytest.fail("bad index should fail before yielding any batch")


def test_raw_shard_size_mismatch_is_rejected(local_data):
    root, _, readers = local_data
    (root / "train" / "shard.mds").write_bytes(b"short")
    readers.append(SimpleNamespace(
        validate=lambda allow_unsafe: None,
        file_pairs=[(SimpleNamespace(basename="shard.mds", bytes=100), None)],
    ))
    with pytest.raises(ValueError, match="shard size"):
        with il.local_well_batches(root, "train"):
            pytest.fail("bad shard should fail before yielding any batch")


class TrackedExecutor:
    instances = []

    def __init__(self, *args, **kwargs):
        self.pending = 0
        self.max_pending = 0
        self.futures = []
        self.shutdown_calls = []
        self.instances.append(self)

    def submit(self, fn, *args):
        owner = self

        class TrackedFuture(Future):
            consumed = False
            cancelled_called = False

            def result(self, *args, **kwargs):
                if not self.consumed:
                    owner.pending -= 1
                    self.consumed = True
                return super().result(*args, **kwargs)

            def cancel(self):
                self.cancelled_called = True
                return super().cancel()

        future = TrackedFuture()
        self.pending += 1
        self.max_pending = max(self.max_pending, self.pending)
        self.futures.append(future)
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, *args, **kwargs):
        self.shutdown_calls.append((args, kwargs))


def test_prefetch_is_bounded_and_cancels_remaining_futures():
    executor = TrackedExecutor()
    samples = il._prefetch(executor, lambda value: value, range(1000), limit=3)
    assert next(samples) == 0
    assert executor.max_pending <= 3
    assert len(executor.futures) <= 3
    samples.close()
    assert all(future.consumed or future.cancelled_called for future in executor.futures)


def test_decode_error_propagates_and_executor_closes(local_data, monkeypatch):
    root, _, _ = local_data
    TrackedExecutor.instances.clear()
    monkeypatch.setattr(il, "ThreadPoolExecutor", TrackedExecutor)

    def decode(sample):
        if sample["well_id"] == "well-2":
            raise OSError("unreadable image")
        return sample["image"]

    monkeypatch.setattr(il, "decode_well", decode)
    with pytest.raises(OSError, match="unreadable image"):
        with il.local_well_batches(root, "train", num_workers=2) as batches:
            list(batches)
    assert TrackedExecutor.instances[-1].shutdown_calls[-1][1] == {
        "wait": True, "cancel_futures": True,
    }


def test_consumer_failure_joins_threads_without_using_tmp(local_data, monkeypatch):
    root, _, _ = local_data

    def forbidden_tmp():
        raise AssertionError("inference loading must not use Streaming/IPC temp directories")

    monkeypatch.setattr(tempfile, "gettempdir", forbidden_tmp)
    with pytest.raises(RuntimeError, match="encoder failed"):
        with il.local_well_batches(root, "train", num_workers=2) as batches:
            next(batches)
            raise RuntimeError("encoder failed")
    assert not any(thread.name.startswith("phenostate-embed") for thread in threading.enumerate())


def test_real_local_mds_round_trip(tmp_path):
    streaming = pytest.importorskip("streaming", reason="real MDS integration needs mosaicml-streaming")
    from io import BytesIO
    from PIL import Image
    from phenostate.image_config import IMAGE_SIZE

    buffer = BytesIO()
    Image.fromarray(np.full((IMAGE_SIZE, IMAGE_SIZE), 128, dtype=np.uint8)).save(buffer, format="PNG")
    columns = {"well_id": "str", **{f"ch{i}": "bytes" for i in range(1, 7)}}
    with streaming.MDSWriter(out=str(tmp_path / "train"), columns=columns, compression=None,
                             hashes=["sha256"]) as writer:
        for index in range(3):
            writer.write({"well_id": f"well-{index}", **{f"ch{i}": buffer.getvalue() for i in range(1, 7)}})
    outputs = []
    for workers in (0, 2):
        with il.local_well_batches(tmp_path, "train", batch_size=2, num_workers=workers,
                                  expected_wells=3) as batches:
            outputs.append(list(batches))
    assert [batch["well_id"] for batch in outputs[0]] == [["well-0", "well-1"], ["well-2"]]
    for left, right in zip(*outputs):
        np.testing.assert_array_equal(left["image"].numpy(), right["image"].numpy())
        np.testing.assert_allclose(left["image"].numpy(), 128 / 255)
