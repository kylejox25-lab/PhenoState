"""Synthetic checks for well-level biological and batch diagnostics."""

import numpy as np
import pandas as pd
import pytest

from phenostate.representation_diagnostics import (
    build_pair_plan,
    control_neighbor_enrichment,
    score_pair_plan,
    validate_inputs,
)


def _metadata(rows):
    """Build already classified diagnostic metadata, without raw-image inputs."""
    records = []
    for index, row in enumerate(rows):
        kind = row.get("kind", "COMPOUND")
        experiment = row.get("experiment_name", "experiment-1")
        plate = row.get("plate", "plate-1")
        treatment = row.get("treatment", "drug-A")
        dose = row.get("dose", 1.0 if kind == "COMPOUND" else np.nan)
        guide = row.get("guide", "")
        perturbation = row.get(
            "perturbation",
            f"compound:{treatment}|dose:{dose:g}" if kind == "COMPOUND" else guide,
        )
        record = {
            "well_id": f"well-{index}",
            "kind": kind,
            "perturbation": perturbation,
            "gene": row.get("gene", ""),
            "guide": guide,
            "dose": dose,
            "batch_id": f"{experiment}/{plate}",
            "experiment_name": experiment,
            "context": f"HUVEC|{kind}",
            "is_control": False,
            "control_family": "vehicle" if kind == "COMPOUND" else "intron",
            "treatment": treatment,
        }
        record.update({key: value for key, value in row.items() if key != "plate"})
        records.append(record)
    return pd.DataFrame(records).reset_index(drop=True)


@pytest.fixture
def crossed_compounds():
    # Two compounds occur on both plates. No controls are present deliberately.
    return _metadata([
        {"treatment": "drug-A", "plate": "plate-1"},
        {"treatment": "drug-B", "plate": "plate-1"},
        {"treatment": "drug-A", "plate": "plate-2"},
        {"treatment": "drug-B", "plate": "plate-2"},
    ])


def _effects(meta, embeddings, plan=None):
    if plan is None:
        plan, _ = build_pair_plan(meta)
    unit = validate_inputs(meta, embeddings)
    return pd.DataFrame(score_pair_plan(unit, meta, plan)["effects"])


def test_biological_vectors_outperform_batch_vectors_on_same_pair_plan(crossed_compounds):
    meta = crossed_compounds
    plan, _ = build_pair_plan(meta)
    original_plan = plan.copy(deep=True)
    original_meta = meta.copy(deep=True)
    biological = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    batch_only = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    effects = _effects(meta, {"biological": biological, "batch_only": batch_only}, plan)
    pd.testing.assert_frame_equal(plan, original_plan)
    pd.testing.assert_frame_equal(meta, original_meta)

    bio = effects.loc[effects["model"].eq("biological")]
    batch = effects.loc[effects["model"].eq("batch_only")]
    assert len(bio) > 0 and len(batch) > 0
    assert bio["replicate_advantage"].notna().all()
    assert batch["replicate_advantage"].notna().all()
    assert bio["competition"].notna().all()
    assert batch["competition"].notna().all()
    np.testing.assert_allclose(bio["replicate_advantage"], 1.0)
    np.testing.assert_allclose(batch["replicate_advantage"], 0.0)
    np.testing.assert_allclose(bio["competition"], 1.0)
    np.testing.assert_allclose(batch["competition"], -1.0)
    np.testing.assert_allclose(bio["plate_advantage"], 0.0)
    np.testing.assert_allclose(batch["plate_advantage"], 1.0)


def test_same_gene_other_guide_is_not_a_nonrepeat_candidate():
    meta = _metadata([
        {"kind": "CRISPR", "gene": "GENE-X", "guide": "guide-x1", "plate": "plate-1"},
        {"kind": "CRISPR", "gene": "GENE-X", "guide": "guide-x2", "plate": "plate-2"},
        {"kind": "CRISPR", "gene": "GENE-X", "guide": "guide-x1", "plate": "plate-2"},
        {"kind": "CRISPR", "gene": "GENE-Y", "guide": "guide-y1", "plate": "plate-2"},
        {"kind": "CRISPR", "gene": "GENE-Y", "guide": "guide-y1", "plate": "plate-1"},
    ])
    plan, _ = build_pair_plan(meta)
    nonrepeat = plan.loc[plan["relation"].str.startswith("nonrepeat")]
    assert not nonrepeat.empty
    left = meta.iloc[nonrepeat["left"].astype(int).to_numpy()]
    right = meta.iloc[nonrepeat["right"].astype(int).to_numpy()]
    assert (left["gene"].to_numpy() != right["gene"].to_numpy()).all()
    repeat = plan.loc[plan["relation"].eq("repeat_cross_plate")]
    assert not repeat.empty
    left = meta.iloc[repeat["left"].astype(int).to_numpy()]
    right = meta.iloc[repeat["right"].astype(int).to_numpy()]
    assert (left["guide"].to_numpy() == right["guide"].to_numpy()).all()


def test_compound_nonrepeat_candidates_match_dose_and_change_compound():
    meta = _metadata([
        {"treatment": treatment, "dose": dose, "plate": plate}
        for treatment, dose in [("drug-A", 0.1), ("drug-B", 0.1), ("drug-B", 1.0)]
        for plate in ("plate-1", "plate-2")
    ])
    plan, _ = build_pair_plan(meta)
    nonrepeat = plan.loc[plan["relation"].str.startswith("nonrepeat")]
    assert not nonrepeat.empty
    left = meta.iloc[nonrepeat["left"].astype(int).to_numpy()]
    right = meta.iloc[nonrepeat["right"].astype(int).to_numpy()]
    np.testing.assert_allclose(left["dose"], right["dose"])
    assert (left["treatment"].to_numpy() != right["treatment"].to_numpy()).all()


def test_missing_controls_preserve_na_instead_of_zero(crossed_compounds):
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    effects = _effects(crossed_compounds, {"model": vectors})
    assert not effects.empty
    assert effects["control_deviation"].isna().all()
    assert effects["replicate_advantage"].notna().all()


def test_zero_norm_embedding_is_rejected(crossed_compounds):
    vectors = np.ones((len(crossed_compounds), 2), dtype=float)
    vectors[1] = 0.0
    with pytest.raises(ValueError):
        validate_inputs(crossed_compounds, {"model": vectors})


def test_constant_control_ties_have_unit_enrichment_and_empirical_baseline():
    # Unequal plate and experiment sizes expose an incorrect 1/n_batches baseline.
    assignments = (
        [("experiment-1", "plate-1")] * 3
        + [("experiment-1", "plate-2")] * 2
        + [("experiment-2", "plate-1")] * 3
        + [("experiment-2", "plate-2")] * 3
    )
    meta = _metadata([
        {"experiment_name": experiment, "plate": plate, "treatment": "EMPTY_control",
         "perturbation": "EMPTY_control", "is_control": True}
        for experiment, plate in assignments
    ])
    vectors = np.tile([1.0, 0.0], (len(meta), 1))
    unit = validate_inputs(meta, {"constant": vectors})
    result = control_neighbor_enrichment(unit, meta, k=2)
    details = pd.DataFrame(result["neighbor_wells"])
    summary = pd.DataFrame(result["neighbors"])
    assert not details.empty
    assert set(details["batch_level"]) == {"plate", "experiment"}
    np.testing.assert_allclose(details["enrichment"], 1.0)
    np.testing.assert_allclose(summary["enrichment"].dropna(), 1.0)

    by_id = meta.set_index("well_id")
    for row in details.itertuples(index=False):
        query = by_id.loc[row.well_id]
        if row.batch_level == "plate":
            pool = meta.loc[meta["experiment_name"].eq(query["experiment_name"])]
            same = pool["batch_id"].eq(query["batch_id"])
        else:
            pool = meta
            same = pool["experiment_name"].eq(query["experiment_name"])
        expected = (int(same.sum()) - 1) / (len(pool) - 1)
        assert row.n_candidates == len(pool) - 1
        assert row.expected == pytest.approx(expected)
        assert row.observed == pytest.approx(expected)
