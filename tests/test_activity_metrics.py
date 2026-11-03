from itertools import permutations

import numpy as np
import pytest

from phenostate.activity_metrics import _expected_ap, activity_metrics


RULES = {"CRISPR": {"where": {"well_type_label": ["Negative controls"]}, "match_fields": ["cell_type"]}}


def dataset(*, experiments=("A",), controls=True):
    rows, values, queries, gallery = [], [], [], []
    for experiment in experiments:
        for plate in range(1, 5):
            for slot in range(5 if controls else 1):
                key = f"{experiment}_{plate}_{slot}"
                treated = slot == 0
                rows.append({"well_id": key, "experiment_name": experiment, "plate": str(plate),
                             "perturbation_type": "CRISPR", "well_type_label": "Query guides" if treated else "Negative controls",
                             "gene": "G" if treated else "", "treatment": "G_guide_1" if treated else "NT",
                             "concentration": "", "cell_type": "HUVEC"})
                values.append([1., 0.] if treated else [0., 1.])
                (queries if plate == 4 else gallery).append(key)
    return np.array(values), rows, queries, gallery


def evaluate(data, **kwargs):
    values, rows, query, gallery = data
    return activity_metrics(values, rows, control_rules=RULES, query_well_ids=query,
                            candidate_well_ids=gallery, n_permutations=199, n_bootstrap=20, **kwargs)


def test_separated_activity_has_high_ap_energy_and_correct_control_audit():
    result = evaluate(dataset())
    summary = result["summary"]
    assert summary["map"] == 1.
    assert summary["energy_distance"] == pytest.approx(2 * np.sqrt(2))
    assert summary["percent_active"] == 100.
    assert summary["percent_energy_active"] == 100.
    assert summary["coverage"] == 1.
    assert summary["by_kind"]["CRISPR"]["n_tests"] == 1
    group = result["groups"][0]
    assert group["perturbation_label"] == "guide:G_guide_1"
    assert group["q_value"] == group["p_value"]
    assert len(result["blocks"][0]["plates"]) == 4
    assert all(plate["n_controls_used"] == 4 for plate in result["blocks"][0]["plates"])
    assert result["queries"][0]["n_positives"] == 3
    assert result["queries"][0]["n_candidates"] == 15


def test_identical_vectors_do_not_produce_energy_activity():
    values, rows, query, gallery = dataset()
    values[:] = [1., 0.]
    result = evaluate((values, rows, query, gallery))
    assert result["summary"]["energy_distance"] == 0.
    assert result["groups"][0]["energy_p_value"] == 1.
    assert result["summary"]["percent_energy_active"] == 0.
    assert result["groups"][0]["p_value"] == 1.
    assert result["summary"]["percent_active"] == 0.
    assert result["queries"][0]["ap"] == pytest.approx(result["queries"][0]["random_ap"])


def test_no_cross_plate_or_cell_type_control_fallback():
    values, rows, query, gallery = dataset()
    for row in rows:
        if row["plate"] == "4" and row["well_type_label"] == "Negative controls":
            row["cell_type"] = "OTHER"
    result = evaluate((values, rows, query, gallery))
    assert result["summary"]["map"] is None
    assert result["summary"]["coverage"] == 0.
    assert result["queries"][0]["reason"] == "no_strictly_matched_controls"
    assert result["blocks"][0]["plates"][-1]["n_controls_used"] == 0


def test_control_rules_cannot_select_queries_or_annotated_positive_controls():
    values, rows, query, gallery = dataset()
    bad = {"CRISPR": {"where": {"well_type_label": ["Query guides"]}}}
    with pytest.raises(ValueError, match="query or annotated positive control"):
        activity_metrics(values, rows, control_rules=bad, query_well_ids=query, candidate_well_ids=gallery,
                         n_permutations=5, n_bootstrap=0)
    rows[1]["well_type_label"] = "CRISPR Gene Positive Controls"
    bad = {"CRISPR": {"where": {"well_type_label": ["CRISPR Gene Positive Controls"]}}}
    with pytest.raises(ValueError, match="positive control"):
        activity_metrics(values, rows, control_rules=bad, query_well_ids=query, candidate_well_ids=gallery,
                         n_permutations=5, n_bootstrap=0)


def test_unique_perturbation_is_tested_once_across_experiments():
    result = evaluate(dataset(experiments=("A", "B")))
    assert len(result["blocks"]) == 2
    assert len(result["groups"]) == 1
    assert result["groups"][0]["n_blocks"] == 2
    assert result["summary"]["n_groups"] == 1
    assert result["summary"]["by_kind"]["CRISPR"]["n_tests"] == 1
    assert result["summary"]["n_evaluable_queries"] == 2


def test_sampling_and_null_are_invariant_to_input_row_order():
    data = dataset(experiments=("A", "B"))
    values, rows, query, gallery = data
    first = evaluate(data, max_controls_per_plate=2)
    second = evaluate((values[::-1], rows[::-1], query, gallery), max_controls_per_plate=2)
    assert first["groups"] == second["groups"]
    assert first["queries"] == second["queries"]
    assert first["blocks"] == second["blocks"]


def test_query_and_gallery_roles_must_be_disjoint_whole_plates():
    values, rows, query, gallery = dataset()
    with pytest.raises(ValueError, match="disjoint"):
        activity_metrics(values, rows, control_rules=RULES, query_well_ids=query,
                         candidate_well_ids=[*gallery, query[0]], n_permutations=5, n_bootstrap=0)
    with pytest.raises(ValueError, match="mix within a plate"):
        activity_metrics(values, rows, control_rules=RULES, query_well_ids=[query[0]],
                         candidate_well_ids=[*gallery, *query[1:]], n_permutations=5, n_bootstrap=0)


def test_missing_control_fields_fail_explicitly():
    values, rows, query, gallery = dataset()
    del rows[0]["cell_type"]
    with pytest.raises(ValueError, match="metadata fields are missing"):
        evaluate((values, rows, query, gallery))


def test_empty_matching_fields_cannot_match_unknown_values():
    values, rows, query, gallery = dataset()
    rows[0]["cell_type"] = ""
    with pytest.raises(ValueError, match="matching metadata fields are empty"):
        evaluate((values, rows, query, gallery))


def test_tied_ap_matches_enumerating_all_rank_orders():
    # Only the middle three candidates tie; one positive precedes that block.
    scores = np.array([2., 1., 1., 1., 0.])
    relevance = np.array([True, False, True, True, False])
    observed = []
    for middle in permutations((1, 2, 3)):
        ordered = relevance[[0, *middle, 4]]
        ranks = np.flatnonzero(ordered) + 1
        observed.append(np.mean(np.arange(1, len(ranks) + 1) / ranks))
    assert float(_expected_ap(scores, relevance)) == pytest.approx(np.mean(observed))
    np.testing.assert_allclose(_expected_ap(np.stack([scores, scores]), relevance), np.mean(observed))
