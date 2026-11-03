"""Export one unnormalized encoder embedding for every well in an MDS split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from phenostate.data.inference_loader import local_well_batches
from phenostate.models import model_from_checkpoint


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_existing_export(output: Path, *, expected: int, tags: dict[bytes, bytes]) -> int:
    """Reuse only a complete native export from the same frozen checkpoint."""
    with pq.ParquetFile(output) as parquet:
        schema = parquet.schema_arrow
        if schema.names != ["well_id", "embedding"] or not pa.types.is_string(schema.field("well_id").type):
            raise ValueError(f"Existing embedding has an unexpected schema: {output}")
        kind = schema.field("embedding").type
        if not pa.types.is_fixed_size_list(kind) or kind.value_type != pa.float32() or kind.list_size < 1:
            raise ValueError(f"Existing embedding must contain fixed float32 vectors: {output}")
        saved_tags = schema.metadata or {}
        if any(saved_tags.get(key) != value for key, value in tags.items()):
            raise ValueError(f"Existing embedding has different or unverified checkpoint/split provenance: {output}")
        if parquet.metadata.num_rows != expected:
            raise ValueError(f"Existing embedding has {parquet.metadata.num_rows} wells; manifest expects {expected}")
        seen: set[str] = set()
        for batch in parquet.iter_batches(batch_size=8192, columns=["well_id", "embedding"]):
            for well_id in batch.column(0).to_pylist():
                if not well_id or well_id in seen:
                    raise ValueError(f"Existing embedding has empty or duplicate well_id: {output}")
                seen.add(well_id)
            vectors = batch.column(1)
            flat = vectors.flatten()
            if vectors.null_count or flat.null_count or not np.isfinite(flat.to_numpy()).all():
                raise ValueError(f"Existing embedding has null or non-finite vectors: {output}")
        if len(seen) != expected:
            raise ValueError(f"Existing embedding contains {len(seen)} wells; manifest expects {expected}")
        return len(seen)


def export_embeddings(
    checkpoint: Path,
    mds_root: Path,
    split: str,
    output: Path,
    *,
    batch_size: int = 1,
    num_workers: int = 4,
    skip_existing: bool = False,
) -> int:
    """Stream embeddings to Parquet and verify one row per expected well."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")
    root, destination = mds_root.resolve(), output.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Embedding output must be outside the MDS directory")
    if output.exists() and not skip_existing:
        raise FileExistsError(f"Output already exists: {output}; use --skip-existing to validate and reuse it")
    temporary = output.with_suffix(output.suffix + ".incomplete")
    if temporary.exists():
        raise FileExistsError(
            f"Incomplete export already exists: {temporary}; choose a new output path "
            "or archive it after the old process exits. Partial exports cannot be resumed."
        )
    manifest_bytes = (mds_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "rxrx3-core-well-mds-v1" or not manifest.get("complete"):
        raise ValueError("Embedding export requires a complete MDS conversion")
    expected = manifest["counts"][split]
    if expected < 1:
        raise ValueError(f"No wells in split {split}")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    checkpoint_hash = _checkpoint_sha256(checkpoint)
    # Keep serialized field names stable so existing embeddings remain reusable.
    provenance = {
        b"cellvit_manifest_sha256": manifest_hash.encode(),
        b"cellvit_checkpoint_sha256": checkpoint_hash.encode(),
        b"cellvit_split": split.encode(),
        b"cellvit_pooling": b"mean",
    }
    if output.exists():
        count = _validate_existing_export(output, expected=expected, tags=provenance)
        print(f"Reusing verified embedding: {output}", file=sys.stderr, flush=True)
        return count

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["manifest_sha256"] != manifest_hash:
        raise ValueError("Checkpoint and MDS manifest do not match")
    model = model_from_checkpoint(state)
    config = model.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = state["run_config"]["amp"] and device.type == "cuda"
    model = model.to(device).eval()
    schema = pa.schema([
        ("well_id", pa.string()),
        ("embedding", pa.list_(pa.float32(), config.encoder_dim)),
    ], metadata={
        b"cellvit_model_type": config.model_type.encode(),
        **provenance,
        b"cellvit_checkpoint": str(checkpoint.resolve()).encode(),
        b"cellvit_loader": b"local-threaded",
    })
    del state  # Export does not need the optimizer/loader/RNG training state.
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with (
        local_well_batches(
            mds_root, split, batch_size=batch_size, num_workers=num_workers,
            expected_wells=expected, pin_memory=device.type == "cuda",
        ) as loader,
        temporary.open("xb") as handle,
        pq.ParquetWriter(handle, schema=schema, compression="zstd") as writer,
    ):
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16,
                    enabled=use_amp,
                ):
                    embedding = model.encode(image).float()
                if not torch.isfinite(embedding).all():
                    raise FloatingPointError("Encoder returned non-finite embeddings")
                ids = list(batch["well_id"])
                for well_id in ids:
                    if not isinstance(well_id, str) or not well_id:
                        raise ValueError(f"Invalid well_id in split: {well_id!r}")
                    if well_id in seen:
                        raise ValueError(f"Duplicate well_id in split: {well_id}")
                    seen.add(well_id)
                table = pa.Table.from_arrays([
                    pa.array(ids, type=pa.string()),
                    pa.array(embedding.cpu().tolist(), type=schema.field("embedding").type),
                ], schema=schema)
                writer.write_table(table)
    if len(seen) != expected:
        raise ValueError(f"Exported {len(seen)} wells; manifest expects {expected}")
    os.replace(temporary, output)
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mds-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Read/decode threads for existing MDS files (default: 4; 0: synchronous). No worker processes.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Validate and reuse a complete export from the same checkpoint; never reuse .incomplete files.",
    )
    args = parser.parse_args()
    count = export_embeddings(
        args.checkpoint, args.mds_root, args.split, args.output,
        batch_size=args.batch_size, num_workers=args.num_workers, skip_existing=args.skip_existing,
    )
    print(json.dumps({"split": args.split, "wells": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
