"""Query-to-reference percent replicating with strictly matched null samples."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from .embedding_metrics import _inputs, _selection


def _has_distinct_assignment(choices: Sequence[Mapping[str, list[int]]]) -> bool:
    """Check slot-to-label matching once before attempting random null draws."""
    assigned: dict[str, int] = {}

    def place(slot: int, visited: set[str]) -> bool:
        for label in sorted(choices[slot]):
            if label in visited:
                continue
            visited.add(label)
            if label not in assigned or place(assigned[label], visited):
                assigned[label] = slot
                return True
        return False

    return all(place(slot, set()) for slot in sorted(range(len(choices)), key=lambda i: len(choices[i])))


def replicate_metrics(
    values: np.ndarray, rows: Sequence[Mapping], *, protocol: str = "compound",
    query_well_ids: Iterable[str] | None = None,
    candidate_well_ids: Iterable[str] | None = None,
    metric: str = "cosine", n_null: int = 500, n_bootstrap: int = 1000,
    seed: int = 2026, max_replicates: int | None = 20, progress=None,
) -> dict:
    """Compare median query-gallery replicate similarity to a matched null.

    A group is (compound+dose, experiment), or (guide, experiment). Only
    cross-plate pairs joining a query well to a reference well are scored.
    Query and reference selections must be disjoint. Every null slot preserves
    its role, plate, perturbation type and dose. Compound null slots use mutually
    distinct other compounds; same-guide null slots use mutually distinct other
    genes, excluding every guide of the observed gene. Null labels are sampled
    uniformly, then a well is sampled within that label. Matching is never relaxed.

    This is an explicit query-gallery PR variant, not an all-pairs PR score.
    Percentages/intervals are on 0..100; groups contain 0..1 fractions.
    """
    if protocol not in {"compound", "same_guide"}:
        raise ValueError("protocol must be compound or same_guide")
    if n_null < 1 or n_bootstrap < 0:
        raise ValueError("n_null must be positive and n_bootstrap nonnegative")
    if max_replicates is not None and max_replicates < 2:
        raise ValueError("max_replicates must be >= 2 or None")
    meta, vectors, valid = _inputs(values, rows, metric)
    query = _selection(meta, query_well_ids)
    gallery = _selection(meta, candidate_well_ids)
    if (query & gallery).any():
        raise ValueError("Query and candidate selections must be disjoint for query-gallery PR")
    kind = "COMPOUND" if protocol == "compound" else "CRISPR"
    eligible = np.array([row["is_query"] and row["perturbation_type"].upper() == kind for row in meta])

    def label(index: int) -> str:
        row = meta[index]
        return row["perturbation_label"] if protocol == "compound" else f"guide:{row['guide']}"

    def null_label(index: int) -> str:
        return meta[index]["treatment"] if protocol == "compound" else meta[index]["gene"]

    def slot_key(index: int, role: str):
        row = meta[index]
        return role, row["batch_id"], kind, row["dose"]

    query_groups, gallery_groups = defaultdict(list), defaultdict(list)
    pools: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in sorted(range(len(meta)), key=lambda i: meta[i]["well_id"]):
        if not eligible[index]:
            continue
        group = label(index), meta[index]["experiment_name"]
        if query[index]:
            query_groups[group].append(index)
        if gallery[index]:
            gallery_groups[group].append(index)
        if valid[index]:
            for role, selected in (("query", query[index]), ("gallery", gallery[index])):
                if selected:
                    pools[slot_key(index, role)][null_label(index)].append(index)

    rng = np.random.default_rng(seed)
    records = []
    groups = sorted(query_groups.items())
    if progress is not None:
        groups = progress.track(groups, f"{protocol} replication groups")
    for (group_label, experiment), requested in groups:
        q = [i for i in requested if valid[i]]
        g = [i for i in gallery_groups[group_label, experiment] if valid[i]]
        record = {"perturbation_label": group_label, "experiment_name": experiment,
                  "gene": meta[requested[0]]["gene"], "reason": None,
                  "n_query_available": len(requested),
                  "n_gallery_available": len(gallery_groups[group_label, experiment]),
                  "n_replicates_available": len(requested) + len(gallery_groups[group_label, experiment])}
        if not q or not g:
            records.append({**record, "reason": "no_valid_query_wells" if not q else "no_reference_replicates"})
            continue
        if max_replicates is not None and len(q) + len(g) > max_replicates:
            # Preserve at least one well of each role when capping repeats.
            mandatory = [int(rng.choice(q)), int(rng.choice(g))]
            remaining = [i for i in q + g if i not in mandatory]
            selected = set(mandatory + rng.choice(remaining, max_replicates - 2, replace=False).tolist())
            q, g = [i for i in q if i in selected], [i for i in g if i in selected]
        indices, roles = q + g, ["query"] * len(q) + ["gallery"] * len(g)
        pair_i, pair_j = [], []
        for a in range(len(q)):
            for b in range(len(q), len(indices)):
                if meta[indices[a]]["batch_id"] != meta[indices[b]]["batch_id"]:
                    pair_i.append(a)
                    pair_j.append(b)
        pair_i, pair_j = np.array(pair_i, dtype=int), np.array(pair_j, dtype=int)
        record.update(n_query_used=len(q), n_gallery_used=len(g), n_replicates_used=len(indices),
                      n_pairs=len(pair_i), query_well_ids_used=[meta[i]["well_id"] for i in q],
                      gallery_well_ids_used=[meta[i]["well_id"] for i in g])
        if not len(pair_i):
            records.append({**record, "reason": "no_cross_plate_query_gallery_pairs"})
            continue

        def median_similarity(sample):
            block = vectors[sample]
            return float(np.median(np.einsum("ij,ij->i", block[pair_i], block[pair_j])))

        observed = median_similarity(indices)
        excluded = null_label(indices[0])
        choices = [{name: wells for name, wells in pools[slot_key(i, role)].items() if name != excluded}
                   for i, role in zip(indices, roles)]
        record["median_similarity"] = observed
        n_distinct = len(set().union(*(set(options) for options in choices)))
        record["n_distinct_null_labels"] = n_distinct
        if any(not options for options in choices):
            records.append({**record, "reason": "no_matched_null_candidates"})
            continue
        if n_distinct < len(indices) or not _has_distinct_assignment(choices):
            records.append({**record, "reason": "insufficient_matched_distinct_null_labels"})
            continue
        null = []
        order = sorted(range(len(choices)), key=lambda slot: len(choices[slot]))
        for _ in range(5 * n_null):
            used, sample = set(), {}
            for slot in order:
                available = sorted(name for name in choices[slot] if name not in used)
                if not available:
                    break
                name = available[int(rng.integers(len(available)))]
                wells = choices[slot][name]
                sample[slot] = wells[int(rng.integers(len(wells)))]
                used.add(name)
            if len(sample) == len(indices):
                null.append(median_similarity([sample[slot] for slot in range(len(indices))]))
                if len(null) == n_null:
                    break
        if len(null) < n_null:
            records.append({**record, "reason": "insufficient_matched_null_draws", "n_null": len(null)})
            continue
        threshold = float(np.quantile(null, .95))
        records.append({**record, "null_q95": threshold, "n_null": n_null,
                        "replicating": bool(observed > threshold),
                        "p_value": (1 + sum(score >= observed for score in null)) / (n_null + 1)})

    by_label = defaultdict(list)
    for record in records:
        if record["reason"] is None:
            by_label[record["perturbation_label"]].append(record)
    groups = [{"perturbation_label": name, "gene": items[0]["gene"],
               "n_experiments": len(items),
               "replicating_fraction": float(np.mean([item["replicating"] for item in items]))}
              for name, items in sorted(by_label.items())]
    fractions = np.array([row["replicating_fraction"] for row in groups])
    interval = None
    if len(fractions) and n_bootstrap:
        boot = [100 * fractions[rng.integers(len(fractions), size=len(fractions))].mean() for _ in range(n_bootstrap)]
        interval = np.quantile(boot, [.025, .975]).tolist()
    return {"summary": {
        "protocol": f"query_gallery_{protocol}_percent_replicating", "metric": metric,
        "percent_replicating": float(100 * fractions.mean()) if len(fractions) else None,
        "ci95": {"percent_replicating": interval} if interval is not None else {},
        "n_groups_available": len(records),
        "n_evaluable_groups": sum(row["reason"] is None for row in records),
        "n_evaluable_perturbations": len(groups),
        "coverage": sum(row["reason"] is None for row in records) / len(records) if records else 0.0,
        "excluded_reasons": dict(Counter(row["reason"] for row in records if row["reason"])),
        "n_invalid_vectors": int((~valid).sum()), "n_null": n_null, "n_bootstrap": n_bootstrap,
        "seed": seed, "max_replicates": max_replicates, "null_quantile": .95,
        "bootstrap_unit": "perturbation_label", "aggregation": "experiments within label, then macro label mean",
        "pair_selection": "cross-plate query-gallery pairs only",
        "null_matching": "exact query/gallery role + plate/type/dose; distinct other genes or compounds",
        "null_sampling": "uniform null label, then uniform well within label; matching never relaxed",
    }, "groups": groups, "replicates": records}
