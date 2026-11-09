"""Paired, metadata-conditioned diagnostics for frozen well embeddings.

This is a descriptive protocol, separate from held-out query/gallery scoring.
Pairs are selected once from metadata, then reused for every representation.
No whole-dataset square similarity matrix is constructed.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from phenostate.embedding_metrics import unit_vectors


PAIR_COLUMNS = ["left", "right", "block", "kind", "perturbation", "context",
                "cluster", "relation", "candidate_batch"]
EFFECTS = ("replicate_advantage", "control_deviation", "competition", "plate_advantage",
           "experiment_replicate_advantage")


def _rng(seed, key):
    digest = hashlib.sha256(json.dumps([seed, key], sort_keys=True).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _pick(values, count, rng):
    values = np.asarray(sorted(values), dtype=int)
    return np.sort(rng.choice(values, min(count, len(values)), replace=False))


def _balanced_anchors(frame, count, rng):
    pools = [rng.permutation(group.index.to_numpy()).tolist()
             for _, group in frame.groupby("batch_id", sort=True)]
    anchors = []
    for offset in range(max(map(len, pools), default=0)):
        for position in rng.permutation(len(pools)):
            if offset < len(pools[position]):
                anchors.append(pools[position][offset])
                if len(anchors) == count:
                    return anchors
    return anchors


def validate_inputs(meta, embeddings):
    """Require aligned rows and common valid, nonzero cosine vectors."""
    required = {"well_id", "kind", "perturbation", "gene", "guide", "dose", "batch_id",
                "experiment_name", "context", "is_control", "control_family"}
    if not required <= set(meta):
        raise ValueError(f"Diagnostic metadata missing: {sorted(required - set(meta))}")
    if not meta.index.equals(pd.RangeIndex(len(meta))) or meta.well_id.duplicated().any():
        raise ValueError("Metadata must have a unique well_id and a consecutive RangeIndex")
    if not embeddings or meta.empty:
        raise ValueError("Need embeddings and nonempty common metadata")
    normalized = {}
    for model, matrix in embeddings.items():
        if len(matrix) != len(meta):
            raise ValueError(f"{model}: embedding rows do not match metadata")
        vectors, valid = unit_vectors(matrix)
        if not valid.all():
            raise ValueError(f"{model}: jointly exclude nonfinite/zero rows before building pairs")
        normalized[model] = vectors
    return normalized


def build_pair_plan(meta, *, seed=2026, max_anchors_per_perturbation=8,
                    max_reference_plates=4, max_pairs_per_relation=24):
    """Construct a fixed bounded pair plan and explicit missing-comparison log.

    Repeated wells use exact guide or compound+dose identities. Nonreplicates
    match dose for compounds and exclude the same gene for CRISPR. B-C and C-D
    use identical negative perturbation identities on both plates, the same
    anchor, and the same reference plate. Medians are computed within blocks;
    blocks are averaged within a perturbation/context before macro summaries.
    """
    if min(max_anchors_per_perturbation, max_reference_plates, max_pairs_per_relation) < 1:
        raise ValueError("Pair limits must be positive")
    records, unavailable = [], []
    treatments = meta[~meta.is_control.astype(bool)]
    controls = meta[meta.is_control.astype(bool)]

    def emit(left, rights, relation, block, row, candidate_batch=""):
        for right in rights:
            if int(left) == int(right):
                continue
            cluster = (str(row.gene) if row.kind == "CRISPR" else str(row.get("treatment", "")))
            records.append({"left": int(left), "right": int(right), "block": block,
                            "kind": row.kind, "perturbation": row.perturbation,
                            "context": row.context, "cluster": cluster or row.perturbation,
                            "relation": relation, "candidate_batch": str(candidate_batch)})

    def missing(row, reason):
        unavailable.append({"well_id": row.well_id, "kind": row.kind,
                            "perturbation": row.perturbation, "context": row.context,
                            "batch_id": row.batch_id, "reason": reason})

    for (kind, context), stratum in treatments.groupby(["kind", "context"], sort=True):
        for perturbation, group in stratum.groupby("perturbation", sort=True):
            rng = _rng(seed, [kind, context, perturbation])
            for left in _balanced_anchors(group, max_anchors_per_perturbation, rng):
                row = meta.loc[left]
                same = group[group.batch_id.eq(row.batch_id)]
                within = _pick(same.index.difference([left]), max_pairs_per_relation, rng)
                emit(left, within, "repeat_same_plate", f"{left}/within", row, row.batch_id)
                eligible = stratum[stratum.perturbation.ne(perturbation)]
                if kind == "CRISPR":
                    eligible = eligible[eligible.gene.ne(row.gene)]
                else:
                    eligible = eligible[eligible.dose.eq(row.dose)]
                    if "treatment" in eligible:
                        eligible = eligible[eligible.treatment.ne(row.get("treatment", ""))]
                local_negative = eligible[eligible.batch_id.eq(row.batch_id)]
                same_experiment = group[group.experiment_name.eq(row.experiment_name)
                                        & group.batch_id.ne(row.batch_id)]
                batches = sorted(same_experiment.batch_id.unique())
                if not batches:
                    missing(row, "no_same_perturbation_other_plate_in_experiment")
                if len(batches) > max_reference_plates:
                    batches = sorted(rng.choice(batches, max_reference_plates, replace=False))
                for batch in batches:
                    block = f"{left}/plate/{batch}"
                    positive = same_experiment[same_experiment.batch_id.eq(batch)]
                    emit(left, _pick(positive.index, max_pairs_per_relation, rng),
                         "repeat_cross_plate", block, row, batch)
                    other = eligible[eligible.batch_id.eq(batch)]
                    common = sorted(set(other.perturbation) & set(local_negative.perturbation))
                    if common:
                        chosen = sorted(rng.choice(common, min(len(common), max_pairs_per_relation), replace=False))
                        # Equal contribution from each nonreplicate identity on both plates.
                        cross_ids = [int(rng.choice(other[other.perturbation.eq(p)].index)) for p in chosen]
                        local_ids = [int(rng.choice(local_negative[local_negative.perturbation.eq(p)].index)) for p in chosen]
                        emit(left, cross_ids, "nonrepeat_cross_plate", block, row, batch)
                        emit(left, local_ids, "nonrepeat_same_plate", block, row, row.batch_id)
                    else:
                        emit(left, _pick(other.index, max_pairs_per_relation, rng),
                             "nonrepeat_cross_plate", block, row, batch)
                        missing(row, "no_shared_negative_identities_for_plate_competition")
                    if other.empty:
                        missing(row, "no_eligible_cross_plate_nonreplicates")

                # Strictly the same identity across experiments, never same-gene/different-guide.
                across = group[group.experiment_name.ne(row.experiment_name)]
                experiments = sorted(across.experiment_name.unique())
                if not experiments:
                    missing(row, "no_exact_perturbation_across_experiments")
                if len(experiments) > max_reference_plates:
                    experiments = sorted(rng.choice(experiments, max_reference_plates, replace=False))
                for experiment in experiments:
                    block = f"{left}/experiment/{experiment}"
                    positive = across[across.experiment_name.eq(experiment)]
                    chosen_positive = _pick(positive.index, max_pairs_per_relation, rng)
                    emit(left, chosen_positive, "repeat_cross_experiment", block, row, experiment)
                    target_batches = set(meta.loc[chosen_positive, "batch_id"])
                    negative = eligible[eligible.batch_id.isin(target_batches)]
                    emit(left, _pick(negative.index, max_pairs_per_relation, rng),
                         "nonrepeat_cross_experiment", block, row, experiment)
                    if negative.empty:
                        missing(row, "no_matched_cross_experiment_nonreplicates")

                matched = controls[controls.kind.eq(kind) & controls.context.eq(context)
                                   & controls.batch_id.eq(row.batch_id)
                                   & controls.control_family.eq(row.control_family)]
                if len(matched) < 2:
                    missing(row, "fewer_than_two_matched_controls_on_plate")
                if len(matched):
                    block = f"{left}/control"
                    selected = _pick(matched.index, max_pairs_per_relation, rng)
                    emit(left, selected, "treated_control", block, row, row.batch_id)
                    pairs = list(itertools.combinations(matched.index, 2))
                    picks = _pick(range(len(pairs)), max_pairs_per_relation, rng)
                    for choice in picks:
                        a, b = pairs[choice]
                        emit(a, [b], "control_control_same_plate", block, row, row.batch_id)

    # Control-only batch shifts, stratified by exact control family and context.
    for (kind, context, family), group in controls.groupby(["kind", "context", "control_family"], sort=True):
        rng = _rng(seed, ["controls", kind, context, family])
        for batch, local in group.groupby("batch_id", sort=True):
            for left in _pick(local.index, min(4, max_anchors_per_perturbation), rng):
                row = meta.loc[left].copy()
                row["perturbation"] = f"control:{kind}:{family}"
                row["gene"] = row["treatment"] = row["perturbation"]
                same = _pick(local.index.difference([left]), max_pairs_per_relation, rng)
                emit(left, same, "control_control_same_plate", f"control/{left}/local", row, batch)
                other = group[group.batch_id.ne(batch) & group.experiment_name.eq(row.experiment_name)]
                for target, target_group in list(other.groupby("batch_id", sort=True))[:max_reference_plates]:
                    emit(left, _pick(target_group.index, max_pairs_per_relation, rng),
                         "control_control_cross_plate", f"control/{left}/{target}", row, target)
                other = group[group.experiment_name.ne(row.experiment_name)]
                for target, target_group in list(other.groupby("experiment_name", sort=True))[:max_reference_plates]:
                    emit(left, _pick(target_group.index, max_pairs_per_relation, rng),
                         "control_control_cross_experiment", f"control/{left}/{target}", row, target)
    return (pd.DataFrame(records, columns=PAIR_COLUMNS),
            pd.DataFrame(unavailable, columns=["well_id", "kind", "perturbation", "context", "batch_id", "reason"])
            .drop_duplicates(ignore_index=True))


def score_pair_plan(embeddings, meta, pair_plan, *, block_size=4096):
    """Evaluate signed cosine on the identical pairs for all supplied models."""
    if pair_plan.empty:
        raise ValueError("No eligible diagnostic pairs; inspect metadata/control coverage")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    embeddings = validate_inputs(meta, embeddings)
    group_keys = ["kind", "perturbation", "context", "cluster"]
    group_results, effect_results, block_results = [], [], []
    left, right = pair_plan.left.to_numpy(), pair_plan.right.to_numpy()
    for model, values in embeddings.items():
        print(f"Pair diagnostics: {model} / {len(pair_plan):,} planned comparisons", flush=True)
        scores = np.empty(len(pair_plan), dtype=np.float32)
        for start in range(0, len(scores), block_size):
            sl = slice(start, start + block_size)
            scores[sl] = np.clip(np.einsum("ij,ij->i", values[left[sl]], values[right[sl]]), -1, 1)
        frame = pair_plan.assign(similarity=scores)
        blocks = frame.groupby([*group_keys, "block", "relation"], sort=True).agg(
            similarity=("similarity", "median"), n_pairs=("similarity", "size")).reset_index()
        block_results.append(blocks.assign(model=model))
        groups = blocks.groupby([*group_keys, "relation"], sort=True).agg(
            similarity=("similarity", "mean"), n_pairs=("n_pairs", "sum"), n_blocks=("block", "size")).reset_index()
        group_results.append(groups.assign(model=model))
        pivot = blocks.pivot(index=[*group_keys, "block"], columns="relation", values="similarity")
        def difference(positive, negative):
            return pivot.get(positive, pd.Series(np.nan, index=pivot.index)) - pivot.get(negative, pd.Series(np.nan, index=pivot.index))
        # CC-TC is d(T,C)-d(C,C'), because d=1-cosine.
        block_effects = pd.DataFrame({
            "replicate_advantage": difference("repeat_cross_plate", "nonrepeat_cross_plate"),
            "control_deviation": difference("control_control_same_plate", "treated_control"),
            "competition": difference("repeat_cross_plate", "nonrepeat_same_plate"),
            "plate_advantage": difference("nonrepeat_same_plate", "nonrepeat_cross_plate"),
            "experiment_replicate_advantage": difference("repeat_cross_experiment", "nonrepeat_cross_experiment"),
        })
        effects = block_effects.groupby(level=group_keys).mean().reset_index()
        effects = effects[~effects.perturbation.str.startswith("control:")]
        effect_results.append(effects.assign(model=model))
    return {"pair_groups": pd.concat(group_results, ignore_index=True),
            "effects": pd.concat(effect_results, ignore_index=True),
            "block_scores": pd.concat(block_results, ignore_index=True)}


def summarize_effects(effects, *, seed=2026, n_bootstrap=500):
    """Cluster bootstrap by gene or compound, preserving correlated doses/guides."""
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    records = []
    for (model, kind), frame in effects.groupby(["model", "kind"], sort=True):
        for metric in EFFECTS:
            valid = frame.dropna(subset=[metric])
            by_cluster = [group[metric].to_numpy() for _, group in valid.groupby("cluster", sort=True)]
            lo = hi = np.nan
            if len(by_cluster) >= 2 and n_bootstrap:
                rng = _rng(seed, [kind, metric])  # identical bootstrap cluster draws for every model
                boot = [np.concatenate([by_cluster[i] for i in rng.integers(len(by_cluster), size=len(by_cluster))]).mean()
                        for _ in range(n_bootstrap)]
                lo, hi = np.quantile(boot, [0.025, 0.975])
            records.append({"model": model, "kind": kind, "metric": metric,
                            "mean": valid[metric].mean(), "ci95_lower": lo, "ci95_upper": hi,
                            "n_groups": len(valid), "n_clusters": len(by_cluster),
                            "n_bootstrap": n_bootstrap, "reason": "" if len(valid) else "no_matched_blocks"})
    return pd.DataFrame(records)


def control_neighbor_enrichment(embeddings, meta, *, k=10, seed=2026,
                                max_per_batch=16, max_per_stratum=1000, block_size=128):
    """Tie-aware conditional batch enrichment on a fixed control-only cohort.

    Plate neighbors are restricted to the same experiment; experiment neighbors
    are restricted to the same control family and biological context. The null
    expectation is (n_batch-1)/(n_candidates-1), excluding self. Summaries weight
    batches equally, then strata equally. A tied k-th neighbor is averaged over
    possible orderings, preventing constant vectors from creating false clusters.
    """
    if min(k, max_per_batch, max_per_stratum, block_size) < 1:
        raise ValueError("Neighbor limits must be positive")
    embeddings = validate_inputs(meta, embeddings)
    controls = meta[meta.is_control.astype(bool)]
    records, skips = [], []
    for level in ("plate", "experiment"):
        fields = ["kind", "context", "control_family"] + (["experiment_name"] if level == "plate" else [])
        label_column = "batch_id" if level == "plate" else "experiment_name"
        for key, group in controls.groupby(fields, sort=True):
            rng = _rng(seed, [level, list(key)])
            ids = []
            for _, batch in group.groupby("batch_id", sort=True):
                ids.extend(_pick(batch.index, max_per_batch, rng))
            # Round-robin over batches preserves representation under the global cap.
            chosen = meta.loc[ids]
            ids = _balanced_anchors(chosen, max_per_stratum, rng)
            cohort = meta.loc[ids]
            labels = cohort[label_column].to_numpy()
            if len(cohort) < 3 or len(set(labels)) < 2:
                skips.append({"kind": key[0], "batch_level": level, "stratum": json.dumps(list(key)),
                              "reason": "fewer_than_two_batches_or_three_controls", "n_wells": len(cohort)})
                continue
            actual_k = min(k, len(cohort) - 1)
            counts = pd.Series(labels).value_counts()
            expected = np.array([(counts[label] - 1) / (len(cohort) - 1) for label in labels])
            for singleton in counts[counts.eq(1)].index:
                skips.append({"kind": key[0], "batch_level": level, "stratum": json.dumps(list(key)),
                              "batch": singleton, "reason": "singleton_control_batch", "n_wells": 1})
            for model, values in embeddings.items():
                selected = values[ids]
                for start in range(0, len(ids), block_size):
                    stop = min(start + block_size, len(ids))
                    scores = np.clip(selected[start:stop] @ selected.T, -1, 1)
                    scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
                    threshold = np.partition(scores, -actual_k, axis=1)[:, -actual_k]
                    above, equal = scores > threshold[:, None], scores == threshold[:, None]
                    same = labels[start:stop, None] == labels[None, :]
                    remaining = actual_k - above.sum(1)
                    observed = ((above & same).sum(1) + remaining * (equal & same).sum(1) / equal.sum(1)) / actual_k
                    for offset, position in enumerate(range(start, stop)):
                        if expected[position] == 0:
                            continue  # singleton labels have no possible same-batch neighbor
                        records.append({"model": model, "kind": key[0], "batch_level": level,
                                        "stratum": json.dumps(list(key)), "well_id": meta.loc[ids[position], "well_id"],
                                        "batch": labels[position], "observed": observed[offset],
                                        "expected": expected[position], "enrichment": observed[offset] / expected[position],
                                        "k": actual_k, "n_candidates": len(cohort) - 1})
    details = pd.DataFrame(records)
    summaries = []
    for model in embeddings:
        for kind in sorted(meta.kind.unique()):
            for level in ("plate", "experiment"):
                subset = details[(details.model == model) & (details.kind == kind) & (details.batch_level == level)] if not details.empty else details
                if subset.empty:
                    value = np.nan
                else:
                    value = subset.groupby(["stratum", "batch"]).enrichment.mean().groupby(level="stratum").mean().mean()
                summaries.append({"model": model, "kind": kind, "batch_level": level,
                                  "enrichment": value, "n_wells": len(subset),
                                  "reason": "no_comparable_control_strata" if subset.empty else ""})
    return {"neighbors": pd.DataFrame(summaries), "neighbor_wells": details,
            "neighbor_exclusions": pd.DataFrame(skips)}
