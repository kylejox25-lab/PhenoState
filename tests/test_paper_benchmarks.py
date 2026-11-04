import csv
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.metrics import average_precision_score, roc_auc_score

from phenostate.paper_benchmarks import SOURCES, _binary_ranking, run_benchmarks


def test_ranking_handles_score_ties():
    scores = np.array([[0.3, 0.7, 0.7, 0.2], [0.4, 0.4, 0.4, 0.4]])
    labels = [1, 1, 0, 0]
    ap, auc = _binary_ranking(scores, n_positive=2)
    np.testing.assert_allclose(ap, [average_precision_score(labels, row) for row in scores])
    np.testing.assert_allclose(auc, [roc_auc_score(labels, row) for row in scores])


def test_paper_benchmarks_use_common_wells_and_report_dose_max(tmp_path):
    rows = []

    def well(kind, experiment, treatment, label, dose=""):
        rows.append({"well_id": f"well-{len(rows)}", "perturbation_type": kind,
                     "experiment_name": experiment, "treatment": treatment,
                     "well_type_label": label, "concentration": dose})

    for _ in range(3):
        well("CRISPR", "gene-001", "EMPTY_control", "")
        well("COMPOUND", "compound-001", "EMPTY_control", "")
    for gene in range(24):
        for guide in range(5):
            well("CRISPR", "gene-001", f"G{gene}_guide_{guide}", "Query guides")
    for dose in (0.1, 1.0):
        for _ in range(3):
            well("COMPOUND", "compound-001", "Drug", "Query Compounds + Intron control", dose)

    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for source in SOURCES:
        (tmp_path / f"{source}.txt").write_text("entity1,entity2\nG0,G1\nG1,G0\nG0,G100\n", encoding="utf-8")
    pd.DataFrame({"treatment": "Drug", "gene_symbol": [f"G{i}" for i in range(24)],
                  "nM_value": [1000, 5000] + [15000] * 22}).to_csv(tmp_path / "compound_gene_interactions.csv", index=False)
    ids = [row["well_id"] for row in rows]
    vectors = np.random.default_rng(42).normal(size=(len(ids), 3))
    first, second, wide = (tmp_path / f"{name}.parquet" for name in ("first", "second", "wide"))
    pq.write_table(pa.table({"well_id": ids[::2], "embedding": vectors[::2].tolist()}), first)
    pq.write_table(pa.table({"well_id": ids[1::2], "embedding": vectors[1::2].tolist()}), second)
    pq.write_table(pa.table({"well_id": ids[1:], **{f"feature_{i}": vectors[1:, i] for i in range(3)}}), wide)

    result = run_benchmarks([("split", first, second), ("wide", wide)], metadata, tmp_path,
                            tmp_path / "results", n_seeds=3)
    protocol = json.loads((result / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["n_common_wells"] == len(ids) - 1
    dti = pd.read_csv(result / "dti_summary.csv")
    assert set(dti["dose"]) == {"0.1", "1.0", "max"}
    assert dti["n_compounds"].eq(1).all()
    assert pd.read_csv(result / "gene_relationships.csv")["n_evaluable"].eq(1).all()
    energy = pd.read_csv(result / "energy_groups.csv")
    assert energy.loc[energy["kind"].eq("COMPOUND"), "n_treated"].tolist() == [6, 6]
    for filename in ("energy_groups.csv", "dti_seeds.csv", "gene_relationships.csv"):
        frame = pd.read_csv(result / filename)
        pd.testing.assert_frame_equal(frame[frame["model"].eq("split")].drop(columns="model").reset_index(drop=True),
                                      frame[frame["model"].eq("wide")].drop(columns="model").reset_index(drop=True))
