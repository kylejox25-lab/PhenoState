import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phenostate.diagnostic_plots import make_projection_coordinates, plot_effects


def test_shared_projection_keeps_both_kinds_and_identical_model_wells():
    rows = []
    for kind in ("CRISPR", "COMPOUND"):
        for perturbation in range(3):
            for plate in (1, 2):
                rows.append({"well_id": f"{kind}-{perturbation}-{plate}", "kind": kind,
                             "perturbation": f"{kind}-{perturbation}", "is_control": False,
                             "batch_id": f"e/{plate}", "experiment_name": "e", "context": "HUVEC"})
    meta = pd.DataFrame(rows)
    rng = np.random.default_rng(6)
    embeddings = {"a": rng.normal(size=(len(meta), 5)), "b": rng.normal(size=(len(meta), 9))}
    coordinates, info = make_projection_coordinates(embeddings, meta, ["a", "b"], subset="shared",
                                                   max_perturbations=2, max_points=100)
    assert info["n_selected"] == 4
    assert set(coordinates.kind) == {"CRISPR", "COMPOUND"}
    assert set(coordinates.loc[coordinates.model.eq("a"), "well_id"]) == set(coordinates.loc[coordinates.model.eq("b"), "well_id"])
    assert np.isfinite(coordinates[["x", "y"]]).all().all()


def test_effect_panels_share_quantitative_limits():
    effects = pd.DataFrame({"model": ["a", "b"], "kind": ["CRISPR", "COMPOUND"],
                            "control_deviation": [0.01, 0.8], "replicate_advantage": [0.02, 0.6]})
    fig = plot_effects(effects, ["a", "b"])
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
    plt.close(fig)


def test_collapsed_representation_reports_no_projection_structure():
    meta = pd.DataFrame({"well_id": ["a", "b", "c"], "kind": ["COMPOUND"] * 3,
                         "batch_id": ["e/1", "e/2", "e/3"]})
    coords, info = make_projection_coordinates({"collapsed": np.ones((3, 4))}, meta,
                                               ["collapsed"], subset="all")
    assert coords.empty
    assert "Constant" in info["models"]["collapsed"]["reason"]
