import numpy as np
import pytest

from cellvit.replicate_metrics import replicate_metrics


def data(kind="compound", n_labels=6, n_gallery=1):
    rows, vectors, query, gallery = [], [], [], []
    for index in range(n_labels):
        for replicate in range(n_gallery + 1):
            key = f"{index}-{replicate}"
            rows.append({"well_id": key, "experiment_name": "exp", "plate": str(replicate + 1),
                         "perturbation_type": "COMPOUND" if kind == "compound" else "CRISPR",
                         "well_type_label": "Query Compounds + Intron control" if kind == "compound" else "Query guides",
                         "treatment": f"drug{index}" if kind == "compound" else f"G{index}_guide_1",
                         "gene": "" if kind == "compound" else f"G{index}",
                         "concentration": "2.5" if kind == "compound" else ""})
            vectors.append(np.eye(n_labels)[index])
            (query if replicate == 0 else gallery).append(key)
    return rows, np.asarray(vectors), query, gallery


@pytest.mark.parametrize("protocol", ["compound", "same_guide"])
def test_ideal_query_gallery_replicates_beat_matched_null(protocol):
    rows, values, query, gallery = data(protocol)
    result = replicate_metrics(values, rows, protocol=protocol, query_well_ids=query,
                               candidate_well_ids=gallery, n_null=20, n_bootstrap=20)
    assert result["summary"]["percent_replicating"] == 100
    assert result["summary"]["ci95"]["percent_replicating"] == [100., 100.]
    assert result["summary"]["n_evaluable_groups"] == 6
    assert all(row["n_pairs"] == 1 and row["null_q95"] == 0 for row in result["replicates"])
    assert all(row["replicating_fraction"] == 1 for row in result["groups"])


def test_insufficient_distinct_null_labels_are_reported_without_relaxation():
    rows, values, query, gallery = data(n_labels=2)
    result = replicate_metrics(values, rows, query_well_ids=query, candidate_well_ids=gallery,
                               n_null=100, n_bootstrap=0)
    assert result["summary"]["percent_replicating"] is None
    assert result["summary"]["coverage"] == 0
    assert result["summary"]["excluded_reasons"] == {"insufficient_matched_distinct_null_labels": 2}


def test_cap_preserves_both_roles_and_pairs_only_query_to_gallery():
    rows, values, query, gallery = data(n_labels=8, n_gallery=4)
    result = replicate_metrics(values, rows, query_well_ids=query, candidate_well_ids=gallery,
                               max_replicates=3, n_null=10, n_bootstrap=0)
    for record in result["replicates"]:
        assert record["n_query_used"] == 1
        assert record["n_gallery_used"] == 2
        assert record["n_replicates_used"] == 3
        assert record["n_pairs"] == 2


def test_dose_and_plate_patterns_are_not_relaxed():
    rows, values, query, gallery = data()
    for row in rows:
        if row["well_id"].startswith("0-"):
            row["concentration"] = "10"
    result = replicate_metrics(values, rows, query_well_ids=["0-0"], candidate_well_ids=gallery,
                               n_null=10, n_bootstrap=0)
    assert result["replicates"][0]["reason"] == "no_matched_null_candidates"


def test_other_guides_of_same_gene_do_not_count_as_distinct_null_genes():
    rows, values, query, gallery = data("same_guide", n_labels=3)
    for row in rows:
        if row["gene"] == "G2":
            row["gene"], row["treatment"] = "G1", "G1_guide_2"
    result = replicate_metrics(values, rows, protocol="same_guide", query_well_ids=query,
                               candidate_well_ids=gallery, n_null=10, n_bootstrap=0)
    assert result["replicates"][0]["reason"] == "insufficient_matched_distinct_null_labels"
    assert result["replicates"][0]["n_distinct_null_labels"] == 1


def test_query_gallery_must_be_disjoint_and_reference_must_exist():
    rows, values, query, gallery = data()
    with pytest.raises(ValueError, match="must be disjoint"):
        replicate_metrics(values, rows, query_well_ids=query, candidate_well_ids=query)
    result = replicate_metrics(values, rows, query_well_ids=["0-0"],
                               candidate_well_ids=[key for key in gallery if not key.startswith("0-")],
                               n_null=10, n_bootstrap=0)
    assert result["replicates"][0]["reason"] == "no_reference_replicates"


def test_identical_vectors_do_not_pass_strict_threshold():
    rows, values, query, gallery = data()
    result = replicate_metrics(np.ones_like(values), rows, query_well_ids=query,
                               candidate_well_ids=gallery, n_null=10, n_bootstrap=0)
    assert result["summary"]["percent_replicating"] == 0


def test_order_is_deterministic_by_well_id():
    rows, values, query, gallery = data(n_labels=7, n_gallery=2)
    options = dict(query_well_ids=query, candidate_well_ids=gallery, max_replicates=2,
                   n_null=10, n_bootstrap=20)
    first = replicate_metrics(values, rows, **options)
    second = replicate_metrics(values[::-1], rows[::-1], **options)
    assert first == second
