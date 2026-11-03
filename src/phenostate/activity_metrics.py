"""Control-matched activity tests for frozen, well-level embeddings.

Controls are selected only by explicit metadata rules and matched within
experiment, plate, perturbation type and configured extra fields. Significance
uses plate-blocked label permutations, then BH correction over unique
perturbations separately for each perturbation type. No classifier is fitted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np

from phenostate.embedding_metrics import _inputs, _ks, _rank_record, _selection


def _text(value) -> str:
    return "" if value is None else str(value)


def _rules(meta, control_rules):
    if not isinstance(control_rules, Mapping) or not control_rules:
        raise ValueError("Activity requires explicit, nonempty control_rules")
    rules, controls = {}, defaultdict(list)
    for kind, rule in control_rules.items():
        if kind not in {"CRISPR", "COMPOUND"} or not isinstance(rule, Mapping):
            raise ValueError("Control rules must map CRISPR/COMPOUND to rule objects")
        where = rule.get("where")
        fields = rule.get("match_fields", [])
        if not isinstance(where, Mapping) or not where:
            raise ValueError(f"{kind}: an explicit, nonempty control 'where' rule is required")
        if not isinstance(fields, (list, tuple)) or any(not isinstance(field, str) for field in fields):
            raise ValueError(f"{kind}: match_fields must be a list of metadata fields")
        selected = {}
        for field, values in where.items():
            values = [values] if isinstance(values, str) else values
            if not isinstance(field, str) or not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{kind}: where must map fields to nonempty lists of exact values")
            selected[field] = {_text(value) for value in values}
        fields = tuple(dict.fromkeys(fields))
        relevant = [row for row in meta if row["perturbation_type"].upper() == kind]
        missing = {field for field in (*selected, *fields) if any(field not in row for row in relevant)}
        if missing:
            raise ValueError(f"{kind}: configured metadata fields are missing: {sorted(missing)}")
        empty = {field for field in fields if any(not _text(row.get(field)).strip() for row in relevant)}
        if empty:
            raise ValueError(f"{kind}: matching metadata fields are empty: {sorted(empty)}")
        rules[kind] = fields
        for index, row in enumerate(meta):
            if row["perturbation_type"].upper() != kind:
                continue
            if all(_text(row.get(field)) in values for field, values in selected.items()):
                if row["is_query"] or "positive" in row.get("well_type_label", "").lower():
                    raise ValueError(f"{kind}: control rule selects a query or annotated positive control: {row['well_id']}")
                controls[kind].append(index)
    return rules, controls


def _rng(seed, key):
    digest = hashlib.sha256(json.dumps([seed, key], sort_keys=True).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _p_value(observed, null):
    return float((1 + np.count_nonzero(null >= observed - 1e-12)) / (len(null) + 1))


def _bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1.)
    return result


def _label(row):
    return f"guide:{row['guide']}" if row["perturbation_type"].upper() == "CRISPR" else row["perturbation_label"]


def _expected_ap(scores, relevance):
    """Average AP over every ordering within exact similarity ties.

    In a tie of n candidates with m positives, each rank is positive with
    probability m/n. Conditional on that event, its expected number of earlier
    positives in the tie is (position-1)*(m-1)/(n-1).
    """
    scores, relevance = np.broadcast_arrays(np.asarray(scores), np.asarray(relevance, dtype=bool))
    order = np.argsort(-scores, axis=-1, kind="stable")
    ordered = np.take_along_axis(scores, order, axis=-1)
    positives = np.take_along_axis(relevance, order, axis=-1)
    positions = np.arange(scores.shape[-1])
    starts = np.concatenate((np.ones_like(ordered[..., :1], dtype=bool), ordered[..., 1:] != ordered[..., :-1]), axis=-1)
    ends = np.concatenate((ordered[..., 1:] != ordered[..., :-1], np.ones_like(ordered[..., :1], dtype=bool)), axis=-1)
    first = np.maximum.accumulate(np.where(starts, positions, 0), axis=-1)
    last = np.minimum.accumulate(np.where(ends, positions, len(positions) - 1)[..., ::-1], axis=-1)[..., ::-1]
    cumulative = np.concatenate((np.zeros_like(ordered[..., :1]), np.cumsum(positives, axis=-1)), axis=-1)
    prior = np.take_along_axis(cumulative, first, axis=-1)
    n_positive = np.take_along_axis(cumulative, last + 1, axis=-1) - prior
    size = last - first + 1
    expected_precision = (prior + 1 + (positions - first) * (n_positive - 1) / np.maximum(size - 1, 1)) / (positions + 1)
    total = positives.sum(axis=-1)
    if (total == 0).any():
        raise ValueError("AP requires at least one positive candidate")
    return ((n_positive / size) * expected_precision).sum(axis=-1) / total


def _block_scores(vectors, meta, plates, *, n_permutations, rng, ks):
    """Compute observed statistics and a paired AP/energy null for one context."""
    indices, layout = [], []
    for plate in plates:
        start = len(indices)
        indices.extend(plate["treated"] + plate["controls"])
        positions = np.arange(start, len(indices))
        layout.append((plate["role"], positions, len(plate["treated"])))
    block = vectors[indices].astype(np.float64)
    similarities = block @ block.T
    squared_norms = np.einsum("ij,ij->i", block, block)
    distances = np.sqrt(np.maximum(squared_norms[:, None] + squared_norms[None, :] - 2 * similarities, 0.))
    np.fill_diagonal(distances, 0.)
    treated = np.zeros((n_permutations + 1, len(indices)), dtype=bool)
    query_pool, gallery, energy = [], [], np.zeros(n_permutations + 1)
    for role, positions, n_treated in layout:
        n_controls = len(positions) - n_treated
        treated[0, positions[:n_treated]] = True
        picks = np.argsort(rng.random((n_permutations, len(positions))), axis=1)[:, :n_treated]
        treated[np.arange(1, n_permutations + 1)[:, None], positions[picks]] = True
        weights = np.where(treated[:, positions], 1. / n_treated, -1. / n_controls)
        energy -= np.einsum("bi,ij,bj->b", weights, distances[np.ix_(positions, positions)], weights,
                            optimize=True) / len(layout)
        (query_pool if role == "query" else gallery).extend(positions.tolist())
    query_pool = np.asarray(query_pool, dtype=int)
    gallery = np.asarray(sorted(gallery, key=lambda i: meta[indices[i]]["well_id"]), dtype=int)
    n_query = int(treated[0, query_pool].sum())
    queries = np.stack([query_pool[mask] for mask in treated[:, query_pool]])
    relevance = treated[:, gallery]
    maps = np.empty(n_permutations + 1)
    # Limit temporary rankings to 64 permutations, rather than allocating a
    # permutations × queries × gallery tensor for the whole run.
    for start in range(0, len(treated), 64):
        stop = min(start + 64, len(treated))
        scores = similarities[queries[start:stop, :, None], gallery[None, None, :]]
        ap = _expected_ap(scores, relevance[start:stop, None, :])
        maps[start:stop] = ap.mean(axis=1)
    query_records = []
    ids = [meta[indices[i]]["well_id"] for i in gallery]
    for position in queries[0]:
        row = meta[indices[position]]
        record = {"well_id": row["well_id"], "perturbation_label": _label(row),
                  "perturbation_type": row["perturbation_type"].upper(),
                  "experiment_name": row["experiment_name"], "reason": None}
        rank = _rank_record(similarities[position, gallery], ids, relevance[0], ks)
        record.update({key: rank[key] for key in ("random_ap", "n_candidates", "n_positives")})
        record["ap"] = float(_expected_ap(similarities[position, gallery], relevance[0]))
        query_records.append(record)
    if len(query_records) != n_query:
        raise RuntimeError("Activity permutation lost its fixed query count")
    return float(maps[0]), maps[1:], float(max(energy[0], 0.)), np.maximum(energy[1:], 0.), query_records


def _summary(groups, *, n_bootstrap, seed):
    fields = ("map", "energy_distance", "percent_active", "percent_energy_active")
    result = {"n_groups": len(groups), "ci95": {}}
    if not groups:
        return {**result, **dict.fromkeys(fields)}
    data = np.array([[group["ap"], group["energy_distance"], 100. * group["active"],
                      100. * group["energy_active"]] for group in groups])
    result.update(zip(fields, data.mean(axis=0).tolist()))
    if n_bootstrap:
        rng = _rng(seed, "bootstrap")
        samples = np.array([data[rng.integers(0, len(data), len(data))].mean(axis=0)
                            for _ in range(n_bootstrap)])
        result["ci95"] = {field: np.quantile(samples[:, i], [.025, .975]).tolist()
                          for i, field in enumerate(fields)}
    return result


def activity_metrics(
    values: np.ndarray, rows: Sequence[Mapping], *, control_rules: Mapping,
    query_well_ids: Iterable[str] | None = None, candidate_well_ids: Iterable[str] | None = None,
    metric: str = "cosine", ks: Sequence[int] = (1, 5, 10), n_permutations: int = 1000,
    n_bootstrap: int = 1000, seed: int = 2026, max_controls_per_plate: int = 16,
    q_threshold: float = .05, progress=None,
) -> dict:
    """Test unique guides and compound+dose treatments against matched controls.

    AP retrieves gallery treatment replicates against their matched controls.
    Energy is the mean biased empirical energy distance over matched plates in
    the normalized-vector geometry. Every null preserves treatment counts per
    plate and query/gallery roles. Experiment/context statistics and permutation
    arrays are averaged per unique perturbation BEFORE p-values and BH. Control
    sampling/permutations depend on metadata and seed, never feature values.
    """
    if n_permutations < 1 or n_bootstrap < 0 or max_controls_per_plate < 1 or not 0 < q_threshold < 1:
        raise ValueError("Invalid permutation/bootstrap count, control cap or FDR threshold")
    ks = _ks(ks)
    meta, vectors, valid = _inputs(values, rows, metric)
    query = _selection(meta, query_well_ids)
    gallery = _selection(meta, candidate_well_ids)
    if (query & gallery).any():
        raise ValueError("Activity query and gallery selections must be disjoint")
    rules, controls = _rules(meta, control_rules)
    roles = {}
    for i, row in enumerate(meta):
        if query[i] or gallery[i]:
            role = "query" if query[i] else "gallery"
            if row["batch_id"] in roles and roles[row["batch_id"]] != role:
                raise ValueError("Activity query/gallery roles must not mix within a plate")
            roles[row["batch_id"]] = role
    pools = defaultdict(list)
    for kind, indices in controls.items():
        fields = rules[kind]
        for i in indices:
            row = meta[i]
            if valid[i] and (query[i] or gallery[i]):
                pools[kind, row["batch_id"], tuple(_text(row.get(field)) for field in fields)].append(i)
    requested = [i for i, row in enumerate(meta) if query[i] and row["is_query"]]
    by_context = defaultdict(lambda: {"query": [], "gallery": []})
    query_records, block_records = [], []
    for i, row in enumerate(meta):
        if not row["is_query"] or not (query[i] or gallery[i]):
            continue
        kind, label = row["perturbation_type"].upper(), _label(row)
        record = {"well_id": row["well_id"], "perturbation_label": label,
                  "perturbation_type": kind, "experiment_name": row["experiment_name"]}
        if kind not in rules or not valid[i]:
            if query[i]:
                query_records.append({**record, "reason": "no_control_rule" if kind not in rules else "invalid_query_vector"})
            continue
        context = tuple(_text(row.get(field)) for field in rules[kind])
        key = kind, label, row["experiment_name"], context
        by_context[key]["query" if query[i] else "gallery"].append(i)
    by_treatment = defaultdict(list)
    for key, indices in by_context.items():
        if indices["query"]:
            by_treatment[key[:2]].append((key, indices))
    groups = []
    treatments = sorted(by_treatment.items())
    if progress is not None:
        treatments = progress.track(treatments, "Activity perturbations")
    for (kind, label), contexts in treatments:
        observed_ap, observed_energy, null_ap, null_energy = [], [], [], []
        valid_query_ids = set()
        for key, indices in sorted(contexts):
            _, _, experiment, context = key
            record = {"perturbation_label": label, "perturbation_type": kind,
                      "experiment_name": experiment, "match_context": dict(zip(rules[kind], context)),
                      "reason": None}
            plates = []
            for batch in sorted({meta[i]["batch_id"] for role in indices.values() for i in role}):
                treated = sorted([i for role in indices.values() for i in role if meta[i]["batch_id"] == batch],
                                 key=lambda i: meta[i]["well_id"])
                available = sorted(pools[kind, batch, context], key=lambda i: meta[i]["well_id"])
                rng = _rng(seed, [key, batch, "controls"])
                chosen = sorted(rng.choice(available, min(len(available), max_controls_per_plate), replace=False).tolist(),
                                key=lambda i: meta[i]["well_id"])
                plates.append({"batch_id": batch, "role": roles[batch], "treated": treated,
                               "controls": chosen, "n_controls_available": len(available)})
            record["plates"] = [{"batch_id": plate["batch_id"], "role": plate["role"],
                                  "treated_well_ids": [meta[i]["well_id"] for i in plate["treated"]],
                                  "control_well_ids": [meta[i]["well_id"] for i in plate["controls"]],
                                  "n_treated": len(plate["treated"]), "n_controls_used": len(plate["controls"]),
                                  "n_controls_available": plate["n_controls_available"]} for plate in plates]
            if not indices["gallery"]:
                record["reason"] = "no_gallery_replicates"
            elif any(not plate["controls"] for plate in plates):
                record["reason"] = "no_strictly_matched_controls"
            if record["reason"]:
                for i in indices["query"]:
                    query_records.append({"well_id": meta[i]["well_id"], "perturbation_label": label,
                                          "perturbation_type": kind, "experiment_name": experiment,
                                          "reason": record["reason"]})
                block_records.append(record)
                continue
            ap, ap_null, energy, energy_null, block_queries = _block_scores(
                vectors, meta, plates, n_permutations=n_permutations, rng=_rng(seed, [key, "permutations"]), ks=ks)
            query_records.extend(block_queries)
            valid_query_ids.update(item["well_id"] for item in block_queries)
            observed_ap.append(ap)
            observed_energy.append(energy)
            null_ap.append(ap_null)
            null_energy.append(energy_null)
            record.update(ap=ap, energy_distance=energy, p_value=_p_value(ap, ap_null),
                          energy_p_value=_p_value(energy, energy_null), n_queries=len(block_queries))
            block_records.append(record)
        if observed_ap:
            ap, energy = float(np.mean(observed_ap)), float(np.mean(observed_energy))
            groups.append({"perturbation_label": label, "perturbation_type": kind, "reason": None,
                           "ap": ap, "map": ap, "energy_distance": energy,
                           "p_value": _p_value(ap, np.mean(null_ap, axis=0)),
                           "energy_p_value": _p_value(energy, np.mean(null_energy, axis=0)),
                           "n_blocks": len(observed_ap), "n_blocks_requested": len(contexts),
                           "n_query_wells": len(valid_query_ids)})
    by_kind = {}
    for kind in sorted({group["perturbation_type"] for group in groups}):
        members = [group for group in groups if group["perturbation_type"] == kind]
        for group, q_value, energy_q in zip(members, _bh([g["p_value"] for g in members]),
                                            _bh([g["energy_p_value"] for g in members])):
            group.update(q_value=float(q_value), energy_q_value=float(energy_q),
                         active=bool(q_value < q_threshold), energy_active=bool(energy_q < q_threshold),
                         active_fraction=float(q_value < q_threshold), energy_active_fraction=float(energy_q < q_threshold))
        by_kind[kind] = {**_summary(members, n_bootstrap=n_bootstrap, seed=seed), "n_tests": len(members)}
    good = [record for record in query_records if record["reason"] is None]
    all_labels = {_label(meta[i]) for i in requested}
    summary = {**_summary(groups, n_bootstrap=n_bootstrap, seed=seed),
               "n_queries": len(requested), "n_evaluable_queries": len({row["well_id"] for row in good}),
               "coverage": len({row["well_id"] for row in good}) / len(requested) if requested else 0.,
               "n_groups_available": len(all_labels), "n_evaluable_groups": len(groups),
               "group_coverage": len(groups) / len(all_labels) if all_labels else 0.,
               "excluded_reasons": dict(Counter(row["reason"] for row in query_records if row["reason"])),
               "block_excluded_reasons": dict(Counter(row["reason"] for row in block_records if row["reason"])),
               "by_kind": by_kind, "metric": metric, "seed": seed, "n_permutations": n_permutations,
               "minimum_p_value": 1. / (n_permutations + 1), "n_bootstrap": n_bootstrap,
               "q_threshold": q_threshold, "max_controls_per_plate": max_controls_per_plate,
               "control_rules": control_rules, "n_invalid_vectors": int((~valid).sum()),
               "ap_tie_policy": "expected AP under uniform ordering within exact similarity ties",
               "energy_geometry": f"{metric}-normalized vectors; mean biased empirical energy distance across plates",
               "activity_significance": "one-sided plate-blocked AP permutation test; BH within perturbation type",
               "energy_significance": "separate one-sided plate-blocked energy permutation test; BH within perturbation type",
               "aggregation": "equal experiment/matched-context means per unique guide or compound+dose; macro perturbation mean",
               "bootstrap_unit": "unique perturbation; fixed observed BH decisions"}
    return {"summary": summary, "queries": sorted(query_records, key=lambda row: row["well_id"]),
            "groups": groups, "blocks": block_records}
