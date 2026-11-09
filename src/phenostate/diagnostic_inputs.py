"""Read a bounded, reproducible common-well cohort for Notebook 03."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from phenostate.activity_metrics import _rules
from phenostate.data.source import load_metadata
from phenostate.embedding_metrics import evaluation_metadata, load_embedding_parquet, unit_vectors


def _rng(seed, key):
    digest = hashlib.sha256(json.dumps([seed, key], sort_keys=True).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _split_assignments(meta, path, declared):
    if path is None:
        entries = [{"well_id": row["well_id"], "split": row["split"]}
                   for row in meta if row.get("split")]
    else:
        path = Path(path)
        if path.suffix.lower() == ".json":
            entries = json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                entries = list(csv.DictReader(handle))
        if not isinstance(entries, list):
            raise ValueError("Saved splits must be a list/table of assignments")
    assignments = {}
    for entry in entries:
        key = entry["well_id"] if "well_id" in entry else (entry["experiment_name"], str(int(entry["plate"])))
        if key in assignments or entry.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Duplicate/invalid split assignment: {key}")
        assignments[key] = entry["split"]
    known_wells = {row["well_id"] for row in meta}
    unknown = {key for key in assignments if isinstance(key, str)} - known_wells
    if unknown:
        raise ValueError(f"Split contains unknown wells: {sorted(unknown)[:3]}")
    result = {}
    plates = {}
    for row in meta:
        key = row["well_id"]
        plate = (row["experiment_name"], str(int(row["plate"])))
        by_well, by_plate = assignments.get(key), assignments.get(plate)
        if by_well and by_plate and by_well != by_plate:
            raise ValueError(f"Conflicting saved split for {key}")
        split = by_well or by_plate or declared.get(key)
        if key in declared and split != declared[key]:
            raise ValueError(f"Saved split conflicts with embedding split tag for {key}")
        if split:
            result[key] = split
            plates.setdefault(plate, set()).add(split)
    if any(len(splits) != 1 for splits in plates.values()):
        raise ValueError("A plate must not appear in multiple splits")
    return result


def load_diagnostic_inputs(model_files, metadata_path, *, splits_path=None, control_rules,
                           selected_splits=("train", "val"), seed=2026,
                           max_perturbations_per_kind=200,
                           max_wells_per_perturbation_plate=3, max_controls_per_plate=32):
    """Align every model by ID, sample by metadata, jointly remove bad vectors.

    No split is generated. Partial well manifests are supported: unassigned wells
    are excluded whenever selected_splits is specified. Without a split file,
    metadata assignments or consistent explicit Parquet split tags are required.
    Limits define an exploratory cohort, not an unbiased full-screen estimate.
    """
    if not model_files or seed < 0 or min(max_perturbations_per_kind,
                                        max_wells_per_perturbation_plate, max_controls_per_plate) < 1:
        raise ValueError("Provide named files and positive sampling limits")
    if selected_splits is not None and (not selected_splits or set(selected_splits) - {"train", "val", "test"}):
        raise ValueError("selected_splits must contain saved train/val/test assignments, or explicitly be None")
    metadata = load_metadata(Path(metadata_path))
    rows = evaluation_metadata(list(metadata.values()))
    match_fields, control_indices = _rules(rows, control_rules)
    controls = {i for indices in control_indices.values() for i in indices}
    common = set(metadata)
    inventories, declared = {}, {}
    for model, files in model_files.items():
        if not files:
            raise ValueError(f"{model}: no embedding files")
        seen, file_ids, source = set(), [], []
        for file in files:
            path = Path(file)
            print(f"Indexing {model}: {path}", flush=True)
            parquet = pq.ParquetFile(path)
            if "well_id" not in parquet.schema_arrow.names:
                raise ValueError(f"{path}: missing well_id")
            keys = []
            for batch in parquet.iter_batches(columns=["well_id"]):
                keys.extend(batch.column(0).to_pylist())
            if any(not isinstance(key, str) or not key for key in keys):
                raise ValueError(f"{path}: missing/invalid well_id")
            if len(keys) != len(set(keys)) or seen.intersection(keys):
                raise ValueError(f"{model}: duplicate well_id within/across files")
            unknown = set(keys) - set(metadata)
            if unknown:
                raise ValueError(f"{path}: unknown metadata wells: {sorted(unknown)[:3]}")
            seen.update(keys)
            tags = parquet.schema_arrow.metadata or {}
            split = tags.get(b"phenostate_split", b"").decode()
            if split:
                if split not in {"train", "val", "test"}:
                    raise ValueError(f"{path}: invalid declared split {split}")
                for key in keys:
                    if key in declared and declared[key] != split:
                        raise ValueError(f"Conflicting model split tags for {key}")
                    declared[key] = split
            file_ids.append((path, set(keys)))
            source.append({"path": str(path.resolve()), "n_rows": len(keys), "bytes": path.stat().st_size})
        inventories[model] = {"ids": seen, "files": file_ids, "source": source}
        common.intersection_update(seen)
    if not common:
        raise ValueError("No common wells across all supplied models")
    split_by_id = _split_assignments(rows, splits_path, declared)
    if selected_splits is not None and not split_by_id:
        raise ValueError("No saved split assignments. Provide SPLITS_PATH, metadata split, or explicitly tagged split exports; no split is generated.")

    prepared = []
    for position, row in enumerate(rows):
        if row["well_id"] not in common:
            continue
        split = split_by_id.get(row["well_id"], "unknown")
        if selected_splits is not None and split not in selected_splits:
            continue
        kind = row["perturbation_type"].upper()
        if kind not in match_fields:
            continue
        is_control = position in controls
        if not is_control and not row["is_query"]:
            continue
        fields = sorted(set(match_fields[kind]) | {"cell_type"})
        context = json.dumps([(field, str(row.get(field, ""))) for field in fields], ensure_ascii=False)
        family = json.dumps([(field, str(row.get(field, ""))) for field in sorted(control_rules[kind]["where"])], ensure_ascii=False) if is_control else ""
        perturbation = (f"guide:{row['guide']}" if kind == "CRISPR" else row["perturbation_label"])
        prepared.append({**row, "kind": kind, "context": context, "split": split,
                         "cell_type": str(row.get("cell_type", "")), "is_control": is_control,
                         "control_family": family, "perturbation": f"control:{family}" if is_control else perturbation,
                         "plate": str(int(row["plate"])),
                         "batch_id": f"{row['experiment_name']}/{int(row['plate'])}"})
    meta = pd.DataFrame(prepared)
    if meta.empty:
        raise ValueError("No common query/control wells in the selected saved splits")
    # One explicit control family is needed for a treated-control comparison.
    # Ambiguous controls remain separate for control-only diagnostics.
    family_map = meta.loc[meta.is_control].groupby(["kind", "context"]).control_family.agg(lambda x: sorted(set(x))).to_dict()
    exclusion_records = []
    for index in meta.index[~meta.is_control]:
        row = meta.loc[index]
        families = family_map.get((row.kind, row.context), [])
        if len(families) == 1:
            meta.at[index, "control_family"] = families[0]
        else:
            meta.at[index, "control_family"] = "<unmatched_or_ambiguous_control_family>"
            exclusion_records.append({"well_id": row.well_id, "model": "all", "reason": "missing_or_ambiguous_control_family"})
    selected = []
    for kind, group in meta[~meta.is_control].groupby("kind", sort=True):
        ranking = group.groupby("perturbation").agg(experiments=("experiment_name", "nunique"), plates=("batch_id", "nunique"))
        rng = _rng(seed, ["perturbations", kind])
        ranking["tie_break"] = rng.random(len(ranking))
        chosen = ranking.sort_values(["experiments", "plates", "tie_break"], ascending=[False, False, True]).head(max_perturbations_per_kind).index
        for key, local in group[group.perturbation.isin(chosen)].groupby(["perturbation", "context", "batch_id"], sort=True):
            indices = local.sort_values("well_id").index.to_numpy()
            selected.extend(_rng(seed, ["treated", kind, *key]).choice(indices, min(len(indices), max_wells_per_perturbation_plate), replace=False))
    for key, local in meta[meta.is_control].groupby(["kind", "context", "control_family", "batch_id"], sort=True):
        indices = local.sort_values("well_id").index.to_numpy()
        selected.extend(_rng(seed, ["controls", *key]).choice(indices, min(len(indices), max_controls_per_plate), replace=False))
    meta = meta.loc[selected].sort_values("well_id").reset_index(drop=True)
    selected_ids = set(meta.well_id)
    arrays, valid_all = {}, np.ones(len(meta), dtype=bool)
    for model, inventory in inventories.items():
        parts, feature_names, width = [], None, None
        ids = []
        for file, file_keys in inventory["files"]:
            requested = selected_ids & file_keys
            if not requested:
                continue
            table = load_embedding_parquet(file, well_ids=requested)
            if feature_names is not None and (feature_names != table.feature_names or width != table.values.shape[1]):
                raise ValueError(f"{model}: inconsistent embedding schemas across files")
            feature_names, width = table.feature_names, table.values.shape[1]
            ids.extend(table.well_ids)
            parts.append(table.values)
        positions = {well: i for i, well in enumerate(ids)}
        values = np.concatenate(parts)[[positions[well] for well in meta.well_id]]
        _, valid = unit_vectors(values)
        for well in meta.loc[~valid, "well_id"]:
            exclusion_records.append({"well_id": well, "model": model, "reason": "nonfinite_or_zero_vector_jointly_removed"})
        valid_all &= valid
        arrays[model] = values
    sampled_count = len(meta)
    meta = meta.loc[valid_all].reset_index(drop=True)
    if meta.empty:
        raise ValueError("All sampled wells are invalid in at least one model")
    arrays = {model: values[valid_all] for model, values in arrays.items()}
    coverage = pd.DataFrame([{"model": model, "n_file_wells": len(info["ids"]), "n_common_before_filters": len(common),
                              "n_sampled": sampled_count, "n_common_valid": len(meta),
                              "dimension": arrays[model].shape[1], "n_files": len(info["files"])}
                             for model, info in inventories.items()])
    report = {"n_metadata": len(metadata), "n_common_before_filters": len(common),
              "n_common_without_saved_split": sum(well not in split_by_id for well in common),
              "n_sampled": sampled_count, "n_jointly_removed": sampled_count - len(meta),
              "n_common_valid": len(meta), "n_treated_valid": int((~meta.is_control).sum()),
              "n_controls_valid": int(meta.is_control.sum()), "selected_splits": selected_splits,
              "sampling": "Metadata coverage priority (experiments, plates); seeded tie breaks; no model score filtering",
              "well_ids_sha256": hashlib.sha256("\n".join(meta.well_id).encode()).hexdigest(),
              "sources": {model: info["source"] for model, info in inventories.items()}}
    return {"meta": meta, "embeddings": arrays, "coverage": coverage,
            "exclusions": pd.DataFrame(exclusion_records, columns=["well_id", "model", "reason"]), "report": report}
