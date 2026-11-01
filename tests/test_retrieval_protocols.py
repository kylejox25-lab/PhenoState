import numpy as np
import pytest

from cellvit.embedding_metrics import _rank_record, _summarize, gene_retrieval_metrics, retrieval_metrics


def gene(key, target, guide, experiment="A", plate=1):
    return {"well_id": key, "gene": target, "treatment": f"{target}_guide_{guide}",
            "experiment_name": experiment, "plate": str(plate), "perturbation_type": "CRISPR",
            "well_type_label": "Query guides", "concentration": ""}


def test_same_guide_excludes_same_gene_other_guides_same_plate_and_other_experiments():
    rows = [gene("q", "G", 1, plate=3), gene("positive", "G", 1),
            gene("ignore-other-guide", "G", 2), gene("negative", "H", 1),
            gene("ignore-same-plate", "G", 1, plate=3), gene("ignore-other-exp", "G", 1, experiment="B")]
    values = np.array([[1., 0.], [0., 1.], [1., 0.], [.8, .6], [1., 0.], [1., 0.]])
    result = retrieval_metrics(values, rows, protocol="same_guide", query_well_ids=["q"],
                               aggregation="gene_guide", n_bootstrap=0)
    assert result["summary"]["map"] == .5
    record = result["queries"][0]
    assert record["n_candidates"] == 2
    assert record["n_positives"] == 1
    assert record["first_positive_rank"] == 2
    assert result["summary"]["n_evaluable_guides"] == 1


def test_same_gene_other_guide_does_not_supply_a_same_guide_positive():
    rows = [gene("q", "G", 1, plate=3), gene("other-guide", "G", 2), gene("negative", "H", 1)]
    result = retrieval_metrics(np.array([[1., 0.], [1., 0.], [0., 1.]]), rows,
                               protocol="same_guide", query_well_ids=["q"], n_bootstrap=0)
    assert result["summary"]["coverage"] == 0
    assert result["queries"][0]["reason"] == "no_cross_plate_positives"


def test_equal_guide_prototypes_are_not_dominated_by_repeated_wells():
    rows = [gene("q", "G", 1), gene("excluded-guide", "G", 1, experiment="B")]
    values = [[1., 0.], [100., 0.]]
    for index in range(9):
        rows.append(gene(f"repeat-{index}", "G", 2, experiment="B"))
        values.append([0., 1.])
    rows.extend([gene("guide-3", "G", 3, experiment="B"), gene("negative", "H", 1, experiment="B")])
    values.extend([[1., 0.], [.5, np.sqrt(.75)]])
    options = dict(query_well_ids=["q"], n_bootstrap=0, strict_prototypes=True)
    well_weighted = gene_retrieval_metrics(np.array(values), rows, prototype_weighting="well", **options)
    guide_weighted = gene_retrieval_metrics(np.array(values), rows, prototype_weighting="guide", **options)
    assert well_weighted["summary"]["mrr"] == .5
    assert guide_weighted["summary"]["mrr"] == 1
    assert guide_weighted["queries"][0]["n_candidates"] == 2


def test_hierarchical_summary_equal_weights_wells_then_guides_then_genes():
    records = []
    for target, guide, well, ranks in [("G", "g1", "a", [1, 2]),
                                        ("G", "g1", "b", [1, 1, 1, 1]),
                                        ("G", "g2", "c", [2]), ("H", "h1", "d", [1])]:
        for rank in ranks:
            score = _rank_record([1., 0.] if rank == 1 else [0., 1.],
                                 ["positive", "negative"], [True, False], (1,))
            records.append({"well_id": well, "guide": guide, "perturbation_label": f"gene:{target}",
                            "reason": None, **score})
    result = _summarize(records, 4, (1,), seed=2026, n_bootstrap=10, aggregation="gene_guide")
    groups = {row["perturbation_label"]: row for row in result["groups"]}
    # G: mean(mean(.75, 1), .5) = .6875; macro mean with H=1 is .84375.
    assert groups["gene:G"]["mrr"] == .6875
    assert groups["gene:G"]["n_query_wells"] == 3
    assert groups["gene:G"]["n_query_guides"] == 2
    assert result["summary"]["mrr"] == .84375
    assert result["summary"]["bootstrap_unit"] == "gene"
    assert result["summary"]["guide_coverage"] == 1


def test_strict_degenerate_prototypes_fail_instead_of_changing_candidate_genes():
    rows = [gene("q", "G", 1), gene("g2", "G", 2, experiment="B"),
            gene("g3", "G", 3, experiment="B"), gene("h", "H", 1, experiment="B")]
    values = np.array([[1., 0.], [1., 0.], [-1., 0.], [0., 1.]])
    with pytest.raises(ValueError, match="Invalid nonempty gene prototypes"):
        gene_retrieval_metrics(values, rows, query_well_ids=["q"], prototype_weighting="guide",
                               strict_prototypes=True, n_bootstrap=0)
    legacy = gene_retrieval_metrics(values, rows, query_well_ids=["q"], n_bootstrap=0)
    assert legacy["summary"]["coverage"] == 0


def test_query_guide_exclusion_happens_before_equal_guide_averaging():
    rows = [gene("q", "G", 1), gene("leak1", "G", 1, experiment="B"),
            gene("leak2", "G", 1, experiment="B"), gene("positive", "G", 2, experiment="B"),
            gene("negative", "H", 1, experiment="B")]
    values = np.array([[1., 0.], [100., 0.], [100., 0.], [0., 1.], [1., 0.]])
    result = gene_retrieval_metrics(values, rows, query_well_ids=["q"], prototype_weighting="guide",
                                   aggregation="gene_guide", strict_prototypes=True, n_bootstrap=0)
    assert result["summary"]["mrr"] == .5
    assert result["summary"]["n_groups"] == 1
