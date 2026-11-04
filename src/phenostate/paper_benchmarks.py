"""RxRx3-core paper benchmarks on frozen, well-level embeddings."""

from __future__ import annotations

import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from phenostate.data.source import load_metadata
from phenostate.embedding_metrics import load_embedding_parquet
from phenostate.progress import Progress
from phenostate.retrieval import _csv, _dump, _sha256

DOSES = (0.0025, 0.01, 0.025, 0.1, 0.25, 1.0, 2.5, 10.0)
SOURCES = ("CORUM", "HuMAP", "Reactome", "StringDB")
CONTROL = "EMPTY_control"
PAPER = "https://arxiv.org/html/2503.20158v2#S4"
EFAAR = "https://github.com/recursionpharma/EFAAR_benchmarking/tree/trunk/RxRx3-core_benchmarks"


def _scan_inputs(inputs, metadata, progress):
    common = set(metadata)
    models = {}
    for name, *files in inputs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in models:
            raise ValueError(f"Expected a unique simple model name, got {name!r}")
        ids, provenance = set(), []
        for file in files:
            path = Path(file).resolve()
            progress(f"{name}: indexing {path}")
            parquet = pq.ParquetFile(path)
            if "well_id" not in parquet.schema_arrow.names:
                raise ValueError(f"{path}: missing well_id")
            for batch in parquet.iter_batches(columns=["well_id"]):
                for key in batch.column(0).to_pylist():
                    if key not in metadata or key in ids:
                        raise ValueError(f"{name}: unknown or duplicate well_id {key!r} in {path}")
                    ids.add(key)
            tags = parquet.schema_arrow.metadata or {}
            provenance.append({
                "path": str(path), "sha256": _sha256(path), "n_rows": parquet.metadata.num_rows,
                **{key.decode(): value.decode() for key, value in tags.items() if key.startswith(b"phenostate_")},
            })
        for tag in ("checkpoint_sha256", "manifest_sha256", "pooling", "model_type"):
            if len({p[f"phenostate_{tag}"] for p in provenance if f"phenostate_{tag}" in p}) > 1:
                raise ValueError(f"{name}: embedding files disagree on {tag}")
        models[name] = {"files": provenance, "n_wells": len(ids)}
        common.intersection_update(ids)
    if not common:
        raise ValueError("No common wells across models")
    return sorted(common), models


def _aligned_embeddings(wells, batches, controls, files):
    position = {well: i for i, well in enumerate(wells)}
    values, schema = None, None
    for file in files:
        if not file["n_common"]:
            continue
        table = load_embedding_parquet(file["path"], well_ids=position)
        width = table.values.shape[1]
        if values is None:
            values = np.empty((len(wells), width), dtype=np.float32)
            schema = table.feature_names
        if width != values.shape[1] or schema != table.feature_names:
            raise ValueError(f"{file['path']}: embedding files for one model must have the same schema")
        if not np.isfinite(table.values).all():
            raise ValueError(f"{file['path']}: nonfinite embeddings among common wells")
        values[[position[well] for well in table.well_ids]] = table.values
        del table
    dimension = values.shape[1]
    pca = PCA().fit(values[controls].astype(np.float64))
    aligned = np.empty((len(wells), pca.n_components_), dtype=np.float32)
    for indices in batches.values():
        batch = pca.transform(values[indices])
        aligned[indices] = StandardScaler().fit(batch[controls[indices]]).transform(batch)
    return aligned, dimension


def _energy_groups(meta):
    query = meta[
        (meta["kind"].eq("CRISPR") & meta["well_type_label"].eq("Query guides"))
        | (meta["kind"].eq("COMPOUND") & meta["well_type_label"].eq("Query Compounds + Intron control")
           & meta["dose"].isin(DOSES))
    ]
    return [(kind, perturbation, group.index.to_numpy())
            for (kind, perturbation), group in query.groupby(["kind", "perturbation"]) if len(group) >= 5]


def _mean_control_distances(values, query, control, block=256):
    reference = values[control]
    return np.concatenate([
        euclidean_distances(values[query[start:start + block]], reference).mean(axis=1, dtype=np.float64)
        for start in range(0, len(query), block)
    ])


def _energy(values, groups, controls):
    query = np.concatenate([indices for _, _, indices in groups])
    cross = _mean_control_distances(values, query, controls)
    within_control = cdist(values[controls], values[controls]).mean()
    rows = []
    for (kind, perturbation, indices), cross_group in zip(
        groups, np.split(cross, np.cumsum([len(item[-1]) for item in groups[:-1]]))
    ):
        treated = values[indices]
        distance = 2 * cross_group.mean() - cdist(treated, treated).mean() - within_control
        rows.append({"kind": kind, "perturbation": perturbation, "n_treated": len(indices),
                     "n_controls": len(controls), "energy": max(0.0, distance)})
    return pd.DataFrame(rows)


def _energy_summary(frame):
    records = []
    for kind, group in [("all", frame), *frame.groupby("kind")]:
        observed, random = group["energy"].to_numpy(), group["random_energy"].to_numpy()
        dispersion, random_dispersion = median_abs_deviation(observed), median_abs_deviation(random)
        denominator = np.hypot(dispersion, random_dispersion)
        records.append({
            "kind": kind, "n_groups": len(group), "energy_mean": observed.mean(), "energy_mad": dispersion,
            "random_mean": random.mean(), "random_mad": random_dispersion,
            "z_score": (observed.mean() - random.mean()) / denominator if denominator else None,
        })
    return pd.DataFrame(records)


def _profiles(meta, values):
    genes = meta["kind"].eq("CRISPR")
    drugs = (meta["kind"].eq("COMPOUND") & meta["dose"].isin(DOSES)
             & ~meta["perturbation"].str.contains("_control", regex=False))
    gene_groups = meta[genes].groupby("perturbation").groups
    drug_groups = meta[drugs].groupby(["perturbation", "dose"]).groups
    if len(gene_groups) < 2 or not drug_groups:
        raise ValueError("Need at least two CRISPR profiles and one compound-dose profile")
    gene_profiles = [values[idx].mean(axis=0, dtype=np.float64) for idx in gene_groups.values()]
    drug_profiles = [values[idx].mean(axis=0, dtype=np.float64) for idx in drug_groups.values()]
    return (pd.DataFrame(gene_profiles, index=list(gene_groups)),
            pd.DataFrame(drug_profiles, index=pd.MultiIndex.from_tuples(list(drug_groups))))


def _gene_relationships(genes, annotations):
    similarities = cosine_similarity(genes)
    background = np.sort(similarities[np.triu_indices(len(genes), k=1)])
    positions = pd.Series(np.arange(len(genes)), index=genes.index)
    rows = []
    for source, labels in annotations.items():
        pairs = labels[labels["entity1"].isin(positions.index) & labels["entity2"].isin(positions.index)]
        scores = similarities[positions.loc[pairs["entity1"]].to_numpy(), positions.loc[pairs["entity2"]].to_numpy()]
        low = np.searchsorted(background, scores, side="right") / len(background) <= 0.05
        high = np.searchsorted(background, scores, side="left") / len(background) >= 0.95
        rows.append({"source": source, "n_annotated": len(labels), "n_evaluable": len(scores),
                     "n_candidate_pairs": len(background), "recall": (low | high).mean() if len(scores) else None,
                     "recall_low": low.mean() if len(scores) else None,
                     "recall_high": high.mean() if len(scores) else None})
    if not any(row["n_evaluable"] for row in rows):
        raise ValueError("No known gene relationships overlap the available gene profiles")
    return pd.DataFrame(rows)


def _binary_ranking(scores, n_positive):
    """Vectorized AP and AUROC with equal scores grouped as in sklearn."""
    order = np.argsort(-scores, axis=-1, kind="stable")
    ranked = np.take_along_axis(scores, order, axis=-1)
    positive = order < n_positive
    count = scores.shape[-1]
    rank = np.arange(count)
    changes = ranked[..., 1:] != ranked[..., :-1]
    shape = (*scores.shape[:-1], 1)
    first = np.maximum.accumulate(np.where(np.concatenate([np.ones(shape, bool), changes], axis=-1), rank, 0), axis=-1)
    last = np.minimum.accumulate(
        np.where(np.concatenate([changes, np.ones(shape, bool)], axis=-1), rank, count - 1)[..., ::-1],
        axis=-1,
    )[..., ::-1]
    true_at_last = np.take_along_axis(positive.cumsum(axis=-1), last, axis=-1)
    ap = (positive * true_at_last / (last + 1)).sum(axis=-1) / n_positive
    negatives = count - n_positive
    cum_neg = np.concatenate([np.zeros(shape, int), (~positive).cumsum(axis=-1)], axis=-1)
    before = np.take_along_axis(cum_neg, first, axis=-1)
    through = last + 1 - true_at_last
    auc = (positive * (negatives - (before + through) / 2)).sum(axis=-1) / (n_positive * negatives)
    return ap, auc


def _drug_targets(genes, drugs, truth, n_seeds, min_negatives):
    truth = truth[truth["gene_symbol"].isin(genes.index) & truth["treatment"].isin(drugs.index.get_level_values(0))]
    pool = set(truth["gene_symbol"])
    similarities = pd.DataFrame(np.abs(cosine_similarity(drugs, genes)), index=drugs.index, columns=genes.index)
    seed_metrics = defaultdict(list)
    random_rng = [np.random.default_rng(np.random.SeedSequence([seed, 1])) for seed in range(n_seeds)]
    for drug, labels in truth.groupby("treatment"):
        positives = sorted(set(labels.loc[labels["nM_value"] <= 1000, "gene_symbol"]))
        excluded = set(labels.loc[labels["nM_value"] <= 10000, "gene_symbol"])
        negatives = sorted(pool - excluded)
        n_negatives = max(2 * len(positives), min_negatives)
        if not positives or len(negatives) < n_negatives:
            continue
        candidates = np.array([
            positives + np.random.default_rng(seed).choice(negatives, n_negatives, replace=False).tolist()
            for seed in range(n_seeds)
        ])
        scores = similarities.loc[drug]
        positions = scores.columns.get_indexer(candidates.ravel()).reshape(candidates.shape)
        values = scores.to_numpy()[:, positions]
        values = np.concatenate([values, values.max(axis=0, keepdims=True)])
        ap, auc = _binary_ranking(values, len(positives))
        baseline = np.array([rng.random(candidates.shape[1]) for rng in random_rng])
        random_ap, random_auc = _binary_ranking(baseline, len(positives))
        for i, dose in enumerate([*(str(x) for x in scores.index), "max"]):
            seed_metrics[dose].append(np.column_stack([ap[i], auc[i], random_ap, random_auc]))
    if not seed_metrics:
        raise ValueError("No compound has sufficient annotated targets and eligible negatives")
    rows = []
    for dose, compounds in seed_metrics.items():
        values = np.stack(compounds)
        for seed, (means, medians) in enumerate(zip(values.mean(axis=0), np.median(values, axis=0))):
            names = ("ap", "auroc", "random_ap", "random_auroc")
            rows.append({"dose": dose, "seed": seed, "n_compounds": len(compounds),
                         **dict(zip(names, means)), **dict(zip((f"{key}_median" for key in names), medians))})
    return pd.DataFrame(rows)


def _dti_summary(frame):
    metrics = [c for c in frame if c not in {"dose", "seed", "n_compounds"}]
    result = frame.groupby("dose", sort=False)[metrics].agg(["mean", lambda x: x.std(ddof=0)])
    result.columns = [f"{metric}_{'mean' if stat == 'mean' else 'std'}" for metric, stat in result.columns]
    result["n_compounds"] = frame.groupby("dose")["n_compounds"].first()
    for metric in ("ap", "auroc"):
        denominator = np.hypot(result[f"{metric}_std"], result[f"random_{metric}_std"])
        result[f"{metric}_z"] = (result[f"{metric}_mean"] - result[f"random_{metric}_mean"]) / denominator.replace(0, np.nan)
    return result.reset_index()


def run_benchmarks(inputs, metadata_path, annotations_dir, output_dir, *,
                   n_seeds=100, min_negatives=20, max_controls=1000, seed=2026):
    if min(n_seeds, min_negatives, max_controls) < 1 or seed < 0 or not inputs or any(len(spec) < 2 for spec in inputs):
        raise ValueError("Provide named embedding files, positive seed/sample counts and a nonnegative energy seed")
    output = Path(output_dir).resolve()
    stage = output.with_name(output.name + ".incomplete")
    if output.exists() or stage.exists():
        raise FileExistsError(f"Choose a new output directory: {output} or {stage} already exists")
    annotations_dir = Path(annotations_dir)
    annotation_files = {source: annotations_dir / f"{source}.txt" for source in SOURCES}
    truth_path = annotations_dir / "compound_gene_interactions.csv"
    annotations = {}
    columns = ["entity1", "entity2"]
    for source, path in annotation_files.items():
        labels = pd.read_csv(path, usecols=columns, dtype=str)[columns].dropna()
        labels = pd.DataFrame(np.sort(labels.to_numpy(), axis=1), columns=columns)
        annotations[source] = labels[labels["entity1"].ne(labels["entity2"])].drop_duplicates()
    truth = pd.read_csv(truth_path, usecols=["treatment", "gene_symbol", "nM_value"]).dropna()
    truth["nM_value"] = pd.to_numeric(truth["nM_value"], errors="raise")
    progress = Progress()
    metadata_path = Path(metadata_path)
    metadata = load_metadata(metadata_path)
    wells, models = _scan_inputs(inputs, metadata, progress)
    meta = pd.DataFrame([metadata[well] for well in wells])
    meta["kind"] = meta["perturbation_type"].str.upper()
    meta["dose"] = pd.to_numeric(meta["concentration"], errors="coerce")
    treatment = meta["treatment"]
    meta["perturbation"] = treatment.where(treatment.str.contains("_control", regex=False), treatment.str.split("_", n=1).str[0])
    batches = {key: group.index.to_numpy() for key, group in meta.groupby("experiment_name")}
    controls = treatment.eq(CONTROL).to_numpy()
    missing = [key for key, idx in batches.items() if controls[idx].sum() < 2]
    if missing:
        raise ValueError(f"PCA-CS needs at least two common {CONTROL} wells per experiment: {missing[:10]}")
    groups = _energy_groups(meta)
    if not groups:
        raise ValueError("No CRISPR or compound perturbations with at least five common wells")
    energy_controls = np.flatnonzero(controls)
    energy_controls = np.sort(np.random.default_rng(seed).choice(
        energy_controls, min(max_controls, len(energy_controls)), replace=False))
    progress(f"{len(wells):,} common wells out of {len(metadata):,}; scoring {len(models)} model(s)")
    stage.mkdir(parents=True)
    energy_control_mask = np.isin(np.arange(len(wells)), energy_controls)
    _csv(stage / "well_manifest.csv", [{"well_id": well, "energy_control": bool(energy_control_mask[i])}
                                      for i, well in enumerate(wells)], ["well_id", "energy_control"])
    all_results = defaultdict(list)
    well_set = set(wells)
    for name, model in models.items():
        progress(f"{name}: PCA-CS")
        for file in model["files"]:
            parquet = pq.ParquetFile(file["path"])
            file["n_common"] = sum(key in well_set for batch in parquet.iter_batches(columns=["well_id"])
                                   for key in batch.column(0).to_pylist())
        values, model["embedding_dimension"] = _aligned_embeddings(wells, batches, controls, model["files"])
        model["aligned_dimension"] = values.shape[1]
        progress(f"{name}: energy distance ({len(groups)} perturbations)")
        energy = _energy(values, groups, energy_controls)
        genes, drugs = _profiles(meta, values)
        random = np.random.default_rng(np.random.SeedSequence([seed, 1])).standard_normal(values.shape, dtype=np.float32)
        del values
        progress(f"{name}: random energy baseline")
        energy["random_energy"] = _energy(random, groups, energy_controls)["energy"]
        del random
        progress(f"{name}: gene relationships and DTI")
        relationships = _gene_relationships(genes, annotations)
        dti = _drug_targets(genes, drugs, truth, n_seeds, min_negatives)
        for key, frame in (("energy_groups", energy), ("energy_summary", _energy_summary(energy)),
                           ("gene_relationships", relationships), ("dti_seeds", dti), ("dti_summary", _dti_summary(dti))):
            all_results[key].append(frame.assign(model=name))
    for key, frames in all_results.items():
        pd.concat(frames, ignore_index=True).to_csv(stage / f"{key}.csv", index=False)
    _dump(stage / "protocol.json", {
        "paper": PAPER, "efaar": EFAAR, "models": models, "n_metadata_wells": len(metadata),
        "n_common_wells": len(wells), "n_common_controls": int(controls.sum()), "n_experiments": len(batches),
        "metadata": {"path": str(metadata_path.resolve()), "sha256": _sha256(metadata_path)},
        "annotations": {p.name: _sha256(p) for p in [truth_path, *annotation_files.values()]},
        "scope": "Pooled common wells in supplied files, without a query/gallery split",
        "alignment": f"PCA on common {CONTROL} wells; center/scale per experiment on those controls",
        "energy": "Biased Euclidean energy; query wells pooled by gene or compound across doses vs sampled pooled EMPTY controls; >=5 treated",
        "energy_max_controls": max_controls, "n_energy_controls": len(energy_controls),
        "energy_baseline": "One seeded standard normal matrix with matched well indices and aligned dimension",
        "energy_seed": seed,
        "gene_relationships": "EFAAR perturbation map; signed cosine; two 5% tails of all distinct gene pairs",
        "dti": "EFAAR perturbation map; absolute cosine; fixed doses and maximum over available doses per pair",
        "dti_positive_max_nm": 1000, "dti_excluded_intermediate_max_nm": 10000,
        "dti_min_negatives": min_negatives, "dti_negative_seeds": n_seeds, "dti_doses_um": DOSES,
    })
    stage.rename(output)
    progress(f"Results written to {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", "--model", action="append", nargs="+", required=True,
                        metavar="NAME_OR_PARQUET", help="Model name followed by one or more Parquet files")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True, help="EFAAR benchmark_annotations directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=100, help="Number of DTI negative sampling seeds")
    parser.add_argument("--min-negatives", type=int, default=20)
    parser.add_argument("--max-controls", type=int, default=1000, help="Maximum pooled controls for energy distance")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for energy controls and random embeddings")
    args = parser.parse_args(argv)
    run_benchmarks(args.embedding, args.metadata, args.annotations_dir, args.output_dir,
                   n_seeds=args.n_seeds, min_negatives=args.min_negatives, max_controls=args.max_controls, seed=args.seed)


if __name__ == "__main__":
    main()
