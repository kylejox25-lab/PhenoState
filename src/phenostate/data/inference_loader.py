"""Read existing local MDS files for inference without worker processes or IPC."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .decode import decode_well


def _prefetch(executor, load, indices, limit):
    """Keep a bounded queue; executor.map submits the entire split on Python 3.11."""
    indices = iter(indices)
    pending = deque()
    try:
        for index in islice(indices, limit):
            pending.append(executor.submit(load, index))
        while pending:
            yield pending.popleft().result()
            for index in islice(indices, 1):
                pending.append(executor.submit(load, index))
    finally:
        for future in pending:
            future.cancel()


def _batches(samples, batch_size, pin_memory):
    samples = iter(samples)
    while batch := list(islice(samples, batch_size)):
        images = torch.from_numpy(np.stack([sample["image"] for sample in batch]))
        if pin_memory:
            images = images.pin_memory()
        yield {"image": images, "well_id": [sample["well_id"] for sample in batch]}


@contextmanager
def local_well_batches(
    mds_root: str | Path,
    split: str,
    *,
    batch_size: int = 1,
    num_workers: int = 4,
    expected_wells: int | None = None,
    pin_memory: bool = False,
) -> Iterator[Iterator[dict]]:
    """Yield ordered batches using bounded read/decode threads (0 = synchronous).

    The project converter writes uncompressed MDS shards with compressed JP2
    payloads. LocalDataset reads these files directly, including over NFS,
    without StreamingDataset's cache locks or DataLoader multiprocessing files.
    The context must remain open while batches are consumed so exceptions also
    close the prefetch iterator and join its threads.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")

    from streaming import LocalDataset

    root = Path(mds_root)
    dataset = LocalDataset(local=str(root), split=split)
    if expected_wells is not None and len(dataset) != expected_wells:
        raise ValueError(f"MDS index contains {len(dataset)} wells; manifest expects {expected_wells}")
    for shard in dataset.shards:
        shard.validate(False)
        for raw_info, _ in shard.file_pairs:
            path = root / split / raw_info.basename
            if path.stat().st_size != raw_info.bytes:
                raise ValueError(f"MDS shard size disagrees with index: {path}")

    def load(index):
        sample = dataset[index]
        return {"image": decode_well(sample), "well_id": sample["well_id"]}

    if num_workers == 0:
        batches = _batches(map(load, range(len(dataset))), batch_size, pin_memory)
        try:
            yield batches
        finally:
            batches.close()
        return

    executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="phenostate-embed")
    samples = _prefetch(executor, load, range(len(dataset)), limit=2 * num_workers)
    batches = _batches(samples, batch_size, pin_memory)
    try:
        yield batches
    finally:
        batches.close()
        samples.close()
        executor.shutdown(wait=True, cancel_futures=True)
