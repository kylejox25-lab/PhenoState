"""Input alignment and eligibility guarantees for Notebook 03 diagnostics."""

import csv

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phenostate.diagnostic_inputs import load_diagnostic_inputs


MODEL_NAMES = (
    "mae", "ca_mae", "diffmae", "cnn_mae", "OpenPhenom", "MAE-L-8", "MAE-G-8",
)
QUERY_LABEL = "Query Compounds + Intron control"
CONTROL_RULES = {
    "COMPOUND": {
        "where": {"treatment": ["EMPTY_control"], "well_type_label": [""]},
        "match_fields": ["cell_type"],
    },
}
METADATA_COLUMNS = (
    "well_id", "experiment_name", "plate", "cell_type", "perturbation_type",
    "well_type_label", "gene", "treatment", "concentration", "split",
)


def _metadata_rows():
    rows = []
    for plate, split in (("1", "train"), ("2", "val")):
        for address, treatment in (
            ("A01", "drug-A"), ("A02", "drug-B"),
            ("B01", "EMPTY_control"), ("B02", "EMPTY_control"),
        ):
            is_control = treatment == "EMPTY_control"
            rows.append({
                "well_id": f"compound-001/{plate}/{address}",
                "experiment_name": "compound-001", "plate": plate,
                "cell_type": "HUVEC", "perturbation_type": "COMPOUND",
                "well_type_label": "" if is_control else QUERY_LABEL,
                "gene": "", "treatment": treatment,
                "concentration": "" if is_control else "1.0", "split": split,
            })
    return rows


def _write_metadata(path, rows, *, include_split=True):
    columns = METADATA_COLUMNS if include_split else METADATA_COLUMNS[:-1]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_embedding(path, ids, values):
    pq.write_table(pa.table({
        "well_id": list(ids),
        "embedding": pa.array(np.asarray(values).tolist(), type=pa.list_(pa.float32())),
    }), path)
    return path


def _inputs(tmp_path, rows=None, *, include_split=True):
    rows = _metadata_rows() if rows is None else rows
    metadata = _write_metadata(tmp_path / "metadata.csv", rows, include_split=include_split)
    ids = [row["well_id"] for row in rows]
    files, expected = {}, {}
    for model_index, name in enumerate(MODEL_NAMES):
        values = np.asarray([
            [index + 1.0, model_index + 0.25, index * 0.1 + 0.5]
            for index in range(len(ids))
        ], dtype=np.float32)
        order = np.random.default_rng(2026 + model_index).permutation(len(ids))
        path = _write_embedding(
            tmp_path / f"{name}.parquet", [ids[index] for index in order], values[order],
        )
        files[name] = [path]
        expected[name] = dict(zip(ids, values))
    return metadata, files, expected


def _load(files, metadata, **kwargs):
    settings = {
        "control_rules": CONTROL_RULES,
        "seed": 2026,
        "max_perturbations_per_kind": 200,
        "max_wells_per_perturbation_plate": 64,
        "max_controls_per_plate": 64,
    }
    settings.update(kwargs)
    return load_diagnostic_inputs(files, metadata, **settings)


def test_seven_models_align_by_well_id_instead_of_parquet_order(tmp_path):
    metadata, files, expected = _inputs(tmp_path)
    result = _load(files, metadata)
    ids = result["meta"]["well_id"].tolist()

    assert len(ids) == len(set(ids)) == 8
    assert set(result["embeddings"]) == set(MODEL_NAMES)
    assert {
        "well_id", "kind", "perturbation", "gene", "guide", "dose",
        "experiment_name", "plate", "batch_id", "split", "cell_type",
        "is_control", "control_family", "context",
    } <= set(result["meta"].columns)
    for name in MODEL_NAMES:
        np.testing.assert_allclose(
            result["embeddings"][name], np.stack([expected[name][well] for well in ids]),
        )


def test_invalid_vector_in_one_model_removes_that_well_from_every_model(tmp_path):
    rows = _metadata_rows()
    metadata, files, expected = _inputs(tmp_path, rows)
    zero_id = rows[0]["well_id"]
    nonfinite_id = rows[2]["well_id"]
    values = np.stack([expected["diffmae"][row["well_id"]] for row in rows])
    values[0] = 0
    values[2, 1] = np.nan
    _write_embedding(files["diffmae"][0], [row["well_id"] for row in rows], values)

    result = _load(files, metadata)
    ids = result["meta"]["well_id"].tolist()
    assert set(ids) == {row["well_id"] for row in rows} - {zero_id, nonfinite_id}
    assert not result["exclusions"].empty
    for name in MODEL_NAMES:
        assert result["embeddings"][name].shape == (6, 3)
        assert np.isfinite(result["embeddings"][name]).all()
        np.testing.assert_allclose(
            result["embeddings"][name], np.stack([expected[name][well] for well in ids]),
        )


def test_duplicate_well_across_files_of_one_model_is_rejected(tmp_path):
    metadata, files, expected = _inputs(tmp_path)
    duplicate_id = _metadata_rows()[0]["well_id"]
    extra = _write_embedding(
        tmp_path / "mae_duplicate.parquet", [duplicate_id], [expected["mae"][duplicate_id]],
    )
    files["mae"].append(extra)

    with pytest.raises(ValueError):
        _load(files, metadata)


def test_partial_well_manifest_retains_only_explicit_train_and_val_assignments(tmp_path):
    rows = _metadata_rows()
    for address, treatment in (("A01", "drug-A"), ("B01", "EMPTY_control")):
        is_control = treatment == "EMPTY_control"
        rows.append({
            **rows[0], "well_id": f"compound-001/3/{address}", "plate": "3",
            "treatment": treatment, "well_type_label": "" if is_control else QUERY_LABEL,
            "concentration": "" if is_control else "1.0", "split": "test",
        })
    metadata, files, _ = _inputs(tmp_path, rows, include_split=False)
    manifest = tmp_path / "well_manifest.csv"
    chosen = [rows[index] for index in (0, 2, 4, 6, 8, 9)]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["well_id", "split"])
        writer.writeheader()
        writer.writerows({"well_id": row["well_id"], "split": row["split"]} for row in chosen)

    result = _load(files, metadata, splits_path=manifest)
    expected_ids = {row["well_id"] for row in chosen if row["split"] in {"train", "val"}}
    assert set(result["meta"]["well_id"]) == expected_ids
    assert set(result["meta"]["split"]) == {"train", "val"}
    assert all(len(values) == len(expected_ids) for values in result["embeddings"].values())


def test_default_selection_requires_saved_splits_or_embedding_split_tags(tmp_path):
    metadata, files, _ = _inputs(tmp_path, include_split=False)
    with pytest.raises(ValueError):
        _load(files, metadata)

    # All supplied wells are permitted only when the caller opts in explicitly.
    result = _load(files, metadata, selected_splits=None)
    assert len(result["meta"]) == 8


def test_context_keeps_cell_types_separate_and_controls_are_explicit(tmp_path):
    rows = _metadata_rows()
    for row in rows:
        if row["plate"] == "2":
            row["cell_type"] = "ARPE19"
    metadata, files, _ = _inputs(tmp_path, rows)
    result = _load(files, metadata)
    meta = result["meta"]

    assert meta.groupby("cell_type")["context"].nunique().eq(1).all()
    assert meta["context"].nunique() == 2
    assert meta["control_family"].notna().all()
    expected_controls = {row["well_id"] for row in rows if row["treatment"] == "EMPTY_control"}
    assert set(meta.loc[meta["is_control"].astype(bool), "well_id"]) == expected_controls


def test_control_rule_cannot_select_query_perturbations(tmp_path):
    metadata, files, _ = _inputs(tmp_path)
    query_as_control = {
        "COMPOUND": {
            "where": {"well_type_label": [QUERY_LABEL]},
            "match_fields": ["cell_type"],
        },
    }
    with pytest.raises(ValueError):
        _load(files, metadata, control_rules=query_as_control)
