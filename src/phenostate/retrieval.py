"""Evaluate well embeddings without loading an encoder or image dataset.

Use --embedding NAME ALL_WELLS.parquet for arbitrary representations, and/or
--model NAME TRAIN.parquet QUERY.parquet for split exports. Only NumPy and
PyArrow are required. Labels define evaluation groups, never encoder inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from phenostate.data.source import load_metadata, make_plate_splits
from phenostate.embedding_metrics import (
    EmbeddingTable, align_embedding_tables, evaluation_metadata,
    gene_retrieval_metrics, load_embedding_parquet, retrieval_metrics, unit_vectors,
)
from phenostate.progress import Progress

TASKS = ("activity", "same-guide", "compound", "cross-guide")
RETRIEVAL_TASKS = ("cross-guide", "same-guide")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, records, fields=None):
    fields = fields or sorted(set().union(*(row.keys() for row in records))) or ["well_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _saved_splits(metadata_path, mds_root=None, splits_path=None):
    metadata = load_metadata(metadata_path)
    manifest, manifest_hash = None, None
    if mds_root is not None:
        manifest_path = mds_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "rxrx3-core-well-mds-v1" or not manifest.get("complete"):
            raise ValueError("Evaluation requires a complete RxRx3-core MDS manifest")
        if manifest.get("metadata_sha256") != _sha256(metadata_path):
            raise ValueError("Metadata CSV does not match the MDS manifest SHA256")
        manifest_hash = _sha256(manifest_path)
        splits_path = mds_root / "plate_splits.json"
    if splits_path is not None:
        if splits_path.suffix.lower() == ".json":
            entries = json.loads(splits_path.read_text(encoding="utf-8"))
        else:
            with splits_path.open(encoding="utf-8-sig", newline="") as handle:
                entries = list(csv.DictReader(handle))
    else:
        if not all(row.get("split") for row in metadata.values()):
            raise ValueError("Provide --splits, --mds-root, or a split column in metadata")
        entries = list(metadata.values())
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise ValueError("Split files must contain a list/table of assignment records")
    assignments = {}
    for row in entries:
        key = row["well_id"] if "well_id" in row else (row["experiment_name"], int(row["plate"]))
        if key in assignments or row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Duplicate or invalid split assignment: {key}")
        assignments[key] = row["split"]
    by_id = {key: assignments[key if key in assignments else (row["experiment_name"], int(row["plate"]))]
             for key, row in metadata.items()}
    plate_splits = defaultdict(set)
    for key, row in metadata.items():
        plate_splits[row["experiment_name"], int(row["plate"])].add(by_id[key])
    if any(len(values) != 1 for values in plate_splits.values()):
        raise ValueError("A plate must not appear in multiple splits")
    if manifest is not None:
        expected = make_plate_splits(metadata, seed=manifest["seed"])
        if assignments != expected:
            raise ValueError("Saved plate splits disagree with the metadata and conversion seed")
        counts = {split: sum(value == split for value in by_id.values()) for split in ("train", "val", "test")}
        if counts != manifest.get("counts"):
            raise ValueError("Metadata split counts disagree with the complete MDS manifest")
    return metadata, by_id, manifest_hash


def _read_model(spec, train_path, query_path, *, metadata, split_by_id, query_split,
                manifest_hash, allow_partial, generic=False, progress=None):
    prefix, separator, label = spec.partition(":")
    name = label if separator else prefix
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Use a simple unique embedding name")
    algorithm = prefix if not generic and (separator or prefix in {"mae", "ca_mae", "diffmae", "cnn_mae"}) else None
    paths = defaultdict(set)
    paths[Path(train_path).resolve()].add("train")
    paths[Path(query_path).resolve()].add(query_split)
    selected = {key for key, split in split_by_id.items() if split in {"train", query_split}}
    tables, provenance = [], []
    for path, roles in paths.items():
        if progress is not None:
            progress(f"{name}: reading {path}")
        parquet = pq.ParquetFile(path)
        if "well_id" not in parquet.schema_arrow.names:
            raise ValueError(f"{path}: missing well_id column")
        tags = {key.decode(): value.decode() for key, value in (parquet.schema_arrow.metadata or {}).items()
                if key.startswith(b"phenostate_")}
        checks = {"phenostate_model_type": algorithm, "phenostate_manifest_sha256": manifest_hash}
        if len(roles) == 1:
            checks["phenostate_split"] = next(iter(roles))
        for key, expected in checks.items():
            if expected is not None and key in tags and tags[key] != expected:
                raise ValueError(f"{path}: {key} is {tags[key]!r}, expected {expected!r}")
        ids = []
        for batch in parquet.iter_batches(columns=["well_id"]):
            ids.extend(batch.column(0).to_pylist())
        if any(not isinstance(key, str) or not key for key in ids):
            raise ValueError(f"{path}: well_id must be a nonempty string")
        if any(key not in metadata for key in ids):
            raise ValueError(f"{path}: unknown metadata well_id")
        if len(roles) == 1 and any(split_by_id[key] not in roles for key in ids):
            wrong = next(key for key in ids if split_by_id[key] not in roles)
            raise ValueError(f"{path}: {wrong} belongs to {split_by_id[wrong]}, not {next(iter(roles))}")
        if "phenostate_split" in tags and any(split_by_id[key] != tags["phenostate_split"] for key in ids):
            raise ValueError(f"{path}: actual wells disagree with the declared split")
        expected_ids = {key for key in selected if split_by_id[key] in roles}
        if not allow_partial and not expected_ids <= set(ids):
            raise ValueError(f"{path}: missing split wells; export complete splits or explicitly use --allow-partial")
        tables.append(load_embedding_parquet(path, well_ids=expected_ids))
        if progress is not None:
            progress(f"{name}: loaded {len(tables[-1].well_ids)} wells; hashing {path.name}")
        provenance.append({"path": str(path), "sha256": _sha256(path), "n_file_rows": len(ids), "schema_metadata": tags})
    first = tables[0]
    if any(table.values.shape[1] != first.values.shape[1] or table.feature_names != first.feature_names for table in tables):
        raise ValueError(f"{name}: train and query embedding schemas differ")
    tags = [item["schema_metadata"] for item in provenance]
    checks = ["phenostate_checkpoint_sha256", "phenostate_pooling", "phenostate_model_type"]
    for key in checks:
        if len({item[key] for item in tags if key in item}) > 1:
            raise ValueError(f"{name}: different checkpoint hashes or export provenance ({key})")
    verification = ("sha256_verified" if all("phenostate_checkpoint_sha256" in item for item in tags)
                    else "unverified; arbitrary embedding source")
    table = EmbeddingTable(tuple(key for item in tables for key in item.well_ids),
                           np.concatenate([item.values for item in tables]), first.feature_names)
    return name, table, {"files": provenance, "checkpoint_verification": verification,
                         "embedding_dimension": table.values.shape[1],
                         "source_kind": "whole embedding table" if generic else "split exports"}


def paired_differences(results: Mapping[str, Mapping], *, seed: int, n_bootstrap: int) -> list[dict]:
    output = []
    for model_a, model_b in itertools.combinations(sorted(results), 2):
        for task in sorted(set(results[model_a]) & set(results[model_b])):
            field = "mrr" if task == "cross-guide" else "ap"
            left = {row["perturbation_label"]: row[field] for row in results[model_a][task]["groups"]}
            right = {row["perturbation_label"]: row[field] for row in results[model_b][task]["groups"]}
            if left.keys() != right.keys():
                raise ValueError("Models have different evaluable perturbations")
            differences = np.array([right[key] - left[key] for key in sorted(left)])
            ci = [None, None]
            if len(differences) and n_bootstrap:
                rng = np.random.default_rng(seed)
                boot = [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(n_bootstrap)]
                ci = np.quantile(boot, [0.025, 0.975]).tolist()
            output.append({"task": task, "metric": "mrr" if field == "mrr" else "map",
                           "model_a": model_a, "model_b": model_b, "n_groups": len(differences),
                           "delta_b_minus_a": float(differences.mean()) if len(differences) else None,
                           "ci95_lower": ci[0], "ci95_upper": ci[1], "n_bootstrap": n_bootstrap, "seed": seed})
    return output


def _score(values, rows, task, options, *, control_rules, n_permutations, n_null, max_controls, max_replicates):
    if task == "activity":
        from phenostate.activity_metrics import activity_metrics
        activity_options = {key: value for key, value in options.items() if key not in {"ks", "block_size"}}
        return activity_metrics(values, rows, control_rules=control_rules, n_permutations=n_permutations,
                                max_controls_per_plate=max_controls, **activity_options)
    if task == "cross-guide":
        result = gene_retrieval_metrics(values, rows, prototype_weighting="guide", strict_prototypes=True,
                                        aggregation="gene_guide", **options)
        result["summary"]["random_mrr"] = result["summary"]["random_ap"]
        return result
    from phenostate.replicate_metrics import replicate_metrics
    protocol = "same_guide" if task == "same-guide" else "compound"
    result = retrieval_metrics(values, rows, protocol=protocol,
                               aggregation="gene_guide" if task == "same-guide" else "label", **options)
    replicate_options = {key: value for key, value in options.items() if key not in {"ks", "block_size"}}
    replication = replicate_metrics(values, rows, protocol=protocol, n_null=n_null,
                                    max_replicates=max_replicates, **replicate_options)
    result["summary"]["percent_replicating"] = replication["summary"]["percent_replicating"]
    result["summary"]["replication_coverage"] = replication["summary"]["coverage"]
    result["summary"]["ci95"]["percent_replicating"] = replication["summary"]["ci95"].get("percent_replicating")
    result["replication"] = replication
    return result


def _write_results(stage, results, transforms):
    summaries, type_summaries = [], []
    for name, tasks in results.items():
        for task, result in tasks.items():
            folder = stage / name / task
            folder.mkdir(parents=True)
            _dump(folder / "metrics.json", result)
            for table in ("queries", "groups", "blocks"):
                if table in result:
                    empty_fields = ["well_id", "reason"] if table == "queries" else ["perturbation_label", "reason"]
                    _csv(folder / f"{table}.csv", result[table], None if result[table] else empty_fields)
            if "replication" in result:
                records = result["replication"]["replicates"]
                _csv(folder / "replicates.csv", records, None if records else ["perturbation_label", "reason"])
            summary = result["summary"]
            primary = "mrr" if task == "cross-guide" else "map"
            interval = summary.get("ci95", {}).get(primary) or [None, None]
            row = {"model": name, "task": task, "primary_metric": primary, "score": summary.get(primary),
                   "ci95_lower": interval[0], "ci95_upper": interval[1]}
            row.update({key: value for key, value in summary.items() if value is None or isinstance(value, (int, float))})
            row["excluded_reasons"] = json.dumps(summary.get("excluded_reasons", {}), sort_keys=True)
            row["candidate_count_range"] = json.dumps(summary.get("candidate_count_range"))
            summaries.append(row)
            for kind, sub in summary.get("by_kind", {}).items():
                type_summaries.append({"model": name, "task": task, "perturbation_type": kind,
                                       **{key: value for key, value in sub.items() if isinstance(value, (str, int, float)) or value is None}})
        if name in transforms:
            np.savez(stage / name / "standardization.npz", **transforms[name])
    leading = ["model", "task", "primary_metric", "score", "ci95_lower", "ci95_upper"]
    _csv(stage / "summary.csv", summaries, leading + sorted(set().union(*(row.keys() for row in summaries)) - set(leading)))
    if type_summaries:
        _csv(stage / "summary_by_type.csv", type_summaries)
    lines = ["# Well embedding evaluation", "", "| Embedding | Task | Metric | Score |", "| --- | --- | --- | --- |"]
    for row in summaries:
        score = "NA" if row["score"] is None else f"{row['score']:.6f}"
        lines.append(f"| {row['model']} | {row['task']} | {row['primary_metric']} | {score} |")
    lines.extend(("", "Query wells come from held-out plates; the reference gallery is train.",
                  "Activity and replication additionally use matched reference wells; see protocol.json.",
                  "Coverage, excluded reasons and permutation resolution must accompany scores."))
    (stage / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def evaluate_embeddings(
    models: Sequence[tuple[str, Path, Path]] = (), *, embeddings: Sequence[tuple[str, Path]] = (),
    metadata_path: Path, output_dir: Path, mds_root: Path | None = None, splits_path: Path | None = None,
    query_split: str = "val", tasks: Sequence[str] = RETRIEVAL_TASKS, metric: str = "cosine",
    preprocessing: str = "raw", ks: Sequence[int] = (1, 5, 10), block_size: int = 64,
    n_bootstrap: int = 1000, seed: int = 2026, drop_nonfinite: bool = False, allow_partial: bool = False,
    control_rules: Mapping | None = None, n_permutations: int = 1000, n_null: int = 500,
    max_controls: int = 16, max_replicates: int = 20, log_every: int = 50,
) -> dict:
    tasks = tuple(dict.fromkeys(tasks))
    if not tasks or set(tasks) - set(TASKS) or query_split not in {"val", "test"}:
        raise ValueError("Invalid tasks or query split")
    if not models and not embeddings:
        raise ValueError("Provide split model exports and/or whole embedding files")
    if "activity" in tasks and not control_rules:
        raise ValueError("Activity requires explicit --control-rules; controls are never inferred")
    if preprocessing not in {"raw", "standardize"} or metric not in {"cosine", "pearson"}:
        raise ValueError("Invalid preprocessing or metric")
    if n_bootstrap < 0 or block_size < 1 or not ks or any(k < 1 for k in ks) or n_permutations < 1 or n_null < 1 or log_every < 0:
        raise ValueError("Invalid bootstrap/permutation count, block size or K")
    progress = Progress(log_every)
    progress(f"Starting CPU evaluation: {len(models) + len(embeddings)} models, tasks={','.join(tasks)}, "
             f"permutations={n_permutations}, null={n_null}, bootstrap={n_bootstrap}")
    destination = output_dir.resolve()
    stage = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or stage.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    if mds_root is not None and (destination == mds_root.resolve() or mds_root.resolve() in destination.parents):
        raise ValueError("Output must be outside the MDS directory")
    progress(f"Reading metadata and validating saved splits: {metadata_path}")
    metadata, split_by_id, manifest_hash = _saved_splits(metadata_path, mds_root, splits_path)
    progress(f"Metadata verified: {len(metadata)} wells; gallery=train, query={query_split}")
    wanted = {key for key, split in split_by_id.items() if split in {"train", query_split}}
    tables, provenance = {}, {}
    sources = [(name, train, query, False) for name, train, query in models]
    sources += [(name, path, path, True) for name, path in embeddings]
    for spec, train, query, generic in sources:
        name, table, info = _read_model(spec, train, query, metadata=metadata, split_by_id=split_by_id,
                                        query_split=query_split, manifest_hash=manifest_hash,
                                        allow_partial=allow_partial, generic=generic, progress=progress)
        if name in tables:
            raise ValueError(f"Duplicate embedding name: {name}")
        tables[name], provenance[name] = table, info
    progress("Aligning common wells and checking feature validity")
    aligned, rows, coverage = align_embedding_tables(tables, metadata, well_ids=wanted, drop_nonfinite=drop_nonfinite)
    train_mask = np.array([split_by_id[row["well_id"]] == "train" for row in rows])
    if not train_mask.any() or train_mask.all():
        raise ValueError("Need common training gallery and query wells")
    transforms = {}
    if preprocessing == "standardize":
        for name, values in aligned.items():
            mean = values[train_mask].mean(axis=0, dtype=np.float64)
            scale = values[train_mask].std(axis=0, dtype=np.float64)
            scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
            aligned[name] = ((values - mean) / scale).astype(np.float32)
            transforms[name] = {"mean": mean, "scale": scale}
    validity = {name: unit_vectors(values, metric)[1] for name, values in aligned.items()}
    common = np.logical_and.reduce(list(validity.values()))
    coverage["invalid_metric_vectors_by_model"] = {name: int((~mask).sum()) for name, mask in validity.items()}
    coverage["dropped_invalid_metric_well_ids"] = [row["well_id"] for row, good in zip(rows, common) if not good]
    rows = [row for row, good in zip(rows, common) if good]
    aligned = {name: values[common] for name, values in aligned.items()}
    query_ids = [row["well_id"] for row in rows if split_by_id[row["well_id"]] == query_split]
    train_ids = [row["well_id"] for row in rows if split_by_id[row["well_id"]] == "train"]
    requested_query = sum(split_by_id[key] == query_split for key in wanted)
    requested_train = len(wanted) - requested_query
    if not query_ids or not train_ids:
        raise ValueError("No common valid query or gallery wells remain")
    progress(f"Common valid wells: gallery={len(train_ids)}, query={len(query_ids)}")
    coverage.update(n_requested_query_wells=requested_query, n_requested_gallery_wells=requested_train,
                    n_common_valid_query_wells=len(query_ids), n_common_valid_gallery_wells=len(train_ids),
                    query_embedding_coverage=len(query_ids) / requested_query,
                    gallery_embedding_coverage=len(train_ids) / requested_train)
    original = evaluation_metadata(list(metadata.values()))
    results = {}
    for name, values in aligned.items():
        options = dict(metric=metric, ks=ks, query_well_ids=query_ids, candidate_well_ids=train_ids,
                       block_size=block_size, n_bootstrap=n_bootstrap, seed=seed, progress=progress)
        results[name] = {}
        for task in tasks:
            progress(f"{name}: starting task {task}")
            result = _score(values, rows, task, options, control_rules=control_rules, n_permutations=n_permutations,
                            n_null=n_null, max_controls=max_controls, max_replicates=max_replicates)
            kind = "COMPOUND" if task == "compound" else "CRISPR" if task != "activity" else None
            requested = [row for row in original if row["is_query"] and split_by_id[row["well_id"]] == query_split
                         and (kind is None or row["perturbation_type"].upper() == kind)]
            n_requested = len(requested)
            result["summary"]["overall_query_coverage"] = result["summary"].get("n_evaluable_queries", 0) / n_requested if n_requested else 0.0
            labels = {f"guide:{row['guide']}" if task == "activity" and row["guide"] else row["perturbation_label"]
                      for row in requested}
            result["summary"]["n_requested_groups_total"] = len(labels)
            result["summary"]["overall_group_coverage"] = result["summary"].get("n_groups", 0) / len(labels) if labels else 0.0
            results[name][task] = result
            progress(f"{name}: completed task {task}; evaluable groups={result['summary'].get('n_groups', 0)}")
    for task in tasks:
        signatures = [sorted((row["well_id"], row.get("candidate_experiment", ""), row.get("reason") or "",
                              row.get("n_candidates", 0), row.get("n_positives", 0)) for row in model[task]["queries"])
                      for model in results.values()]
        if any(item != signatures[0] for item in signatures[1:]):
            raise ValueError(f"{task}: candidate eligibility differs between models")
    progress("Computing paired differences and preparing result files")
    paired = paired_differences(results, seed=seed, n_bootstrap=n_bootstrap)
    protocol = {"format": "phenostate-well-evaluation-v2", "query_split": query_split, "gallery_split": "train",
                "query_unit": "single well", "tasks": list(tasks), "metric": metric, "preprocessing": preprocessing,
                "ks": list(ks), "seed": seed, "n_bootstrap": n_bootstrap, "n_null": n_null,
                "n_permutations": n_permutations, "max_controls_per_plate": max_controls, "max_replicates": max_replicates,
                "control_rules": control_rules, "encoder_input": "image only; perturbation metadata is used for evaluation",
                "split_scope": "evaluation wells only; upstream pretraining overlap is not audited",
                "metadata_sha256": _sha256(metadata_path), "manifest_sha256": manifest_hash,
                "split_source": str(mds_root or splits_path or "metadata split column"),
                "split_sha256": _sha256(mds_root / "plate_splits.json" if mds_root is not None else splits_path or metadata_path),
                "models": provenance,
                "allow_partial": allow_partial, "drop_nonfinite": drop_nonfinite,
                "uncertainty": "perturbation bootstrap conditional on supplied embeddings; not training-seed uncertainty",
                "external_functional_truth": "not used",
                "software": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__}}
    progress(f"Writing results: {destination}")
    stage.mkdir(parents=True)
    _dump(stage / "protocol.json", protocol)
    _dump(stage / "coverage.json", coverage)
    summaries = _write_results(stage, results, transforms)
    _csv(stage / "paired_differences.csv", paired,
         ("task", "metric", "model_a", "model_b", "n_groups", "delta_b_minus_a", "ci95_lower", "ci95_upper", "n_bootstrap", "seed"))
    train_set = set(train_ids)
    _csv(stage / "well_manifest.csv", [{**row, "split": split_by_id[row["well_id"]],
                                       "role": "gallery" if row["well_id"] in train_set else "query"} for row in rows],
         ("well_id", "experiment_name", "plate", "gene", "treatment", "split", "role"))
    stage.rename(destination)
    progress(f"Evaluation complete: {destination}")
    return {"output_dir": str(destination), "summary": summaries, "coverage": coverage}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", nargs=3, metavar=("NAME", "TRAIN_PARQUET", "QUERY_PARQUET"))
    parser.add_argument("--embedding", action="append", nargs=2, metavar=("NAME", "PARQUET"))
    parser.add_argument("--metadata", type=Path, required=True)
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--mds-root", type=Path, help="Optional native manifest verification")
    split.add_argument("--splits", type=Path, help="Plate or well split JSON/CSV; otherwise use metadata.split")
    parser.add_argument("--query-split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--control-rules", type=Path, help="Explicit negative-control JSON; required for activity")
    parser.add_argument("--metric", choices=("cosine", "pearson"), default="cosine")
    parser.add_argument("--preprocessing", choices=("raw", "standardize"), default="raw")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50, help="Progress interval in task groups/blocks; 0 disables logs")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--max-controls", type=int, default=16)
    parser.add_argument("--max-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--drop-nonfinite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = evaluate_embeddings(
            args.model or (), embeddings=args.embedding or (), metadata_path=args.metadata,
            mds_root=args.mds_root, splits_path=args.splits, output_dir=args.output_dir, query_split=args.query_split,
            tasks=args.tasks, control_rules=json.loads(args.control_rules.read_text(encoding="utf-8")) if args.control_rules else None,
            metric=args.metric, preprocessing=args.preprocessing, ks=args.ks, block_size=args.block_size,
            n_bootstrap=args.n_bootstrap, n_permutations=args.n_permutations, n_null=args.n_null,
            max_controls=args.max_controls, max_replicates=args.max_replicates, log_every=args.log_every, seed=args.seed,
            drop_nonfinite=args.drop_nonfinite, allow_partial=args.allow_partial)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    print(json.dumps({"output_dir": report["output_dir"], "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
