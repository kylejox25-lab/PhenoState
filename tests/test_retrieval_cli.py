import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cellvit.data.source import make_plate_splits
from cellvit.retrieval import evaluate_embeddings, main


def dataset(tmp_path):
    rows = []
    for experiment in ("gene-A", "gene-B"):
        for plate in (1, 2, 3):
            for gene in ("G", "H"):
                for guide in (1, 2):
                    rows.append({"well_id": f"{experiment}_{plate}_{gene}_{guide}",
                                 "experiment_name": experiment, "plate": str(plate),
                                 "gene": gene, "treatment": f"{gene}_guide_{guide}",
                                 "perturbation_type": "CRISPR", "well_type_label": "Query guides",
                                 "concentration": ""})
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assignments = make_plate_splits({row["well_id"]: row for row in rows})
    by_split = {split: [row for row in rows if assignments[row["experiment_name"], int(row["plate"])] == split]
                for split in ("train", "val", "test")}
    root = tmp_path / "mds"
    root.mkdir()
    manifest = {"format": "rxrx3-core-well-mds-v1", "complete": True, "seed": 2026,
                "counts": {split: len(items) for split, items in by_split.items()},
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "plate_splits.json").write_text(json.dumps([
        {"experiment_name": experiment, "plate": plate, "split": split}
        for (experiment, plate), split in sorted(assignments.items())]), encoding="utf-8")
    manifest_hash = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    models = []
    for algorithm in ("mae", "diffmae"):
        paths = []
        for split in ("train", "val"):
            path = tmp_path / f"{algorithm}_{split}.parquet"
            vectors = [[1., 0.] if row["gene"] == "G" else [0., 1.] for row in by_split[split]]
            if algorithm == "diffmae" and split == "val":
                vectors = [vector[::-1] for vector in vectors]
            table = pa.table({"well_id": [row["well_id"] for row in by_split[split]],
                              "embedding": pa.array(vectors, type=pa.list_(pa.float32(), 2))})
            table = table.replace_schema_metadata({
                b"cellvit_model_type": algorithm.encode(), b"cellvit_pooling": b"mean",
                b"cellvit_manifest_sha256": manifest_hash.encode(), b"cellvit_split": split.encode(),
                b"cellvit_checkpoint": f"/old-server/{algorithm}.pt".encode(),
                b"cellvit_checkpoint_sha256": hashlib.sha256(algorithm.encode()).hexdigest().encode()})
            pq.write_table(table, path)
            paths.append(path)
        models.append((algorithm, *paths))
    return metadata, root, models


def run(data, output, **kwargs):
    metadata, root, models = data
    return evaluate_embeddings(models, metadata_path=metadata, mds_root=root,
                               output_dir=output, n_bootstrap=20, **kwargs)


def rewrite(path, *, values=None, keep=None, tags=None):
    table = pq.read_table(path)
    if keep is not None:
        table = table.take(pa.array(keep, type=pa.int64()))
    if values is not None:
        index = table.schema.get_field_index("embedding")
        table = table.set_column(index, table.schema.field(index),
                                 pa.array(values, type=table.schema.field(index).type))
    if tags is not None:
        table = table.replace_schema_metadata({**table.schema.metadata, **tags})
    pq.write_table(table, path)


def test_end_to_end_two_models_share_train_gallery_and_paired_genes(tmp_path):
    data = dataset(tmp_path)
    output = tmp_path / "results"
    report = run(data, output)
    scores = {(row["model"], row["task"]): row["score"] for row in report["summary"]}
    assert scores["mae", "cross-guide"] == 1.0
    assert scores["diffmae", "cross-guide"] == .5
    assert scores["mae", "same-guide"] == 1.0
    assert report["coverage"]["query_embedding_coverage"] == 1
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["gallery_split"] == "train"
    assert protocol["models"]["mae"]["checkpoint_verification"] == "sha256_verified"
    assert "not used" == protocol["external_functional_truth"]
    with (output / "well_manifest.csv").open(newline="", encoding="utf-8") as handle:
        wells = list(csv.DictReader(handle))
    assert all(row["split"] == ("train" if row["role"] == "gallery" else "val") for row in wells)
    with (output / "paired_differences.csv").open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    cross = next(row for row in paired if row["task"] == "cross-guide")
    # Alphabetical order: diffmae -> mae.
    assert cross["model_a"] == "diffmae"
    assert float(cross["delta_b_minus_a"]) == .5
    assert float(cross["ci95_lower"]) == float(cross["ci95_upper"]) == .5
    assert (output / "mae" / "cross-guide" / "queries.csv").is_file()
    assert not output.with_name("results.incomplete").exists()
    with pytest.raises(FileExistsError):
        run(data, output)


def test_actual_wrong_split_is_rejected_even_without_schema_tags(tmp_path):
    data = dataset(tmp_path)
    train_path, query_path = data[2][0][1:]
    query = pq.read_table(query_path).replace_schema_metadata(None)
    pq.write_table(query, train_path)
    with pytest.raises(ValueError, match="belongs to val, not train"):
        run(data, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_manifest_and_saved_split_integrity_are_checked(tmp_path):
    data = dataset(tmp_path)
    path = data[1] / "plate_splits.json"
    splits = json.loads(path.read_text(encoding="utf-8"))
    splits[0]["split"] = "test" if splits[0]["split"] != "test" else "train"
    path.write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(ValueError, match="Saved plate splits disagree"):
        run(data, tmp_path / "bad")


def test_missing_export_wells_require_explicit_partial_mode_and_report_coverage(tmp_path):
    data = dataset(tmp_path)
    rewrite(data[2][0][2], keep=list(range(7)))
    with pytest.raises(ValueError, match="allow-partial"):
        run(data, tmp_path / "strict")
    report = run(data, tmp_path / "partial", allow_partial=True)
    assert report["coverage"]["n_requested_query_wells"] == 8
    assert report["coverage"]["n_common_valid_query_wells"] == 7
    assert report["coverage"]["query_embedding_coverage"] == 7 / 8
    assert {row["n_queries"] for row in report["summary"]} == {7}


@pytest.mark.parametrize("bad", [[0., 0.], [float("nan"), 0.]])
def test_invalid_vectors_are_removed_from_every_model_together(tmp_path, bad):
    data = dataset(tmp_path)
    path = data[2][0][2]
    values = pq.read_table(path)["embedding"].to_pylist()
    values[0] = bad
    rewrite(path, values=values)
    if np.isnan(bad[0]):
        with pytest.raises(ValueError, match="Nonfinite"):
            run(data, tmp_path / "strict")
    report = run(data, tmp_path / "clean", drop_nonfinite=True)
    assert report["coverage"]["n_common_valid_query_wells"] == 7
    assert {row["n_queries"] for row in report["summary"]} == {7}


def test_standardization_is_fitted_only_on_training_wells(tmp_path):
    data = dataset(tmp_path)
    path = data[2][0][2]
    values = np.array(pq.read_table(path)["embedding"].to_pylist()) * 100
    rewrite(path, values=values.tolist())
    output = tmp_path / "scaled"
    run(data, output, preprocessing="standardize")
    transform = np.load(output / "mae" / "standardization.npz")
    np.testing.assert_allclose(transform["mean"], [.5, .5])
    np.testing.assert_allclose(transform["scale"], [.5, .5])


def test_different_checkpoint_hashes_are_rejected(tmp_path):
    data = dataset(tmp_path)
    rewrite(data[2][0][2], tags={b"cellvit_checkpoint_sha256": b"different"})
    with pytest.raises(ValueError, match="different checkpoint hashes"):
        run(data, tmp_path / "bad")


def test_legacy_checkpoint_paths_need_not_exist_and_model_labels_support_ablations(tmp_path):
    data = dataset(tmp_path)
    name, train, query = data[2][0]
    for path in (train, query):
        table = pq.read_table(path)
        tags = dict(table.schema.metadata)
        del tags[b"cellvit_checkpoint_sha256"]
        pq.write_table(table.replace_schema_metadata(tags), path)
    data = (data[0], data[1], [("mae:seed1", train, query), ("mae:seed2", train, query)])
    output = tmp_path / "aliases"
    report = run(data, output)
    assert {row["model"] for row in report["summary"]} == {"seed1", "seed2"}
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["models"]["seed1"]["checkpoint_verification"].startswith("legacy_path_only")


def test_cli_accepts_explicit_algorithm_embedding_paths_and_output_directory(tmp_path, capsys):
    metadata, root, models = dataset(tmp_path)
    name, train, query = models[0]
    output = tmp_path / "cli"
    main(["--model", name, str(train), str(query), "--metadata", str(metadata),
          "--mds-root", str(root), "--output-dir", str(output), "--n-bootstrap", "0",
          "--tasks", "cross-guide", "same-guide"])
    message = json.loads(capsys.readouterr().out)
    assert message["output_dir"] == str(output.resolve())
    assert len(message["summary"]) == 2


def test_cli_help_renders_repeated_model_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "TRAIN_PARQUET" in capsys.readouterr().out


def test_arbitrary_wide_embeddings_run_without_mds_or_known_algorithm(tmp_path):
    metadata, root, models = dataset(tmp_path)
    source = pa.concat_tables([pq.read_table(path).replace_schema_metadata(None) for path in models[0][1:]])
    vectors = np.array(source["embedding"].to_pylist())
    path = tmp_path / "external_features.parquet"
    pq.write_table(pa.table({"well_id": source["well_id"], "feature_0": vectors[:, 0],
                             "feature_1": vectors[:, 1]}), path)
    output = tmp_path / "independent"
    report = evaluate_embeddings(embeddings=[("any_encoder", path)], metadata_path=metadata,
                                 splits_path=root / "plate_splits.json", output_dir=output,
                                 tasks=["cross-guide"], n_bootstrap=0)
    assert report["summary"][0]["score"] == 1
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["manifest_sha256"] is None
    assert protocol["models"]["any_encoder"]["checkpoint_verification"].startswith("unverified")


def test_metadata_split_column_is_enough_for_independent_evaluation(tmp_path):
    metadata, root, models = dataset(tmp_path)
    with metadata.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = make_plate_splits({row["well_id"]: row for row in rows})
    for row in rows:
        row["split"] = splits[row["experiment_name"], int(row["plate"])]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = evaluate_embeddings(models[:1], metadata_path=metadata, output_dir=tmp_path / "metadata_split",
                                 tasks=["cross-guide"], n_bootstrap=0)
    assert report["summary"][0]["score"] == 1


def test_activity_requires_explicit_control_rules(tmp_path):
    data = dataset(tmp_path)
    with pytest.raises(ValueError, match="explicit --control-rules"):
        run(data, tmp_path / "activity", tasks=["activity"])


def test_well_split_file_cannot_split_a_plate(tmp_path):
    metadata, root, models = dataset(tmp_path)
    with metadata.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = make_plate_splits({row["well_id"]: row for row in rows})
    assignments = [{"well_id": row["well_id"], "split": splits[row["experiment_name"], int(row["plate"])]} for row in rows]
    assignments[0]["split"] = "val" if assignments[0]["split"] != "val" else "train"
    path = tmp_path / "bad_splits.json"
    path.write_text(json.dumps(assignments), encoding="utf-8")
    with pytest.raises(ValueError, match="plate must not appear"):
        evaluate_embeddings(models[:1], metadata_path=metadata, splits_path=path,
                            output_dir=tmp_path / "bad", tasks=["cross-guide"])


def test_all_tasks_on_independent_embeddings_with_controls_and_different_dimensions(tmp_path):
    metadata, _, _ = dataset(tmp_path)
    with metadata.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["cell_type"] = "HUVEC"
    plates = sorted({(row["experiment_name"], row["plate"]) for row in rows})
    for experiment, plate in plates:
        template = {"experiment_name": experiment, "plate": plate, "gene": "", "cell_type": "HUVEC"}
        for index in range(3):
            rows.append({**template, "well_id": f"{experiment}_{plate}_drug{index}",
                         "treatment": f"drug{index}", "concentration": "1.0", "perturbation_type": "COMPOUND",
                         "well_type_label": "Query Compounds + Intron control"})
        for kind in ("CRISPR", "COMPOUND"):
            for index in range(7):
                rows.append({**template, "well_id": f"{experiment}_{plate}_{kind}_control{index}",
                             "treatment": f"intron{index}" if kind == "CRISPR" else "EMPTY_control",
                             "concentration": "", "perturbation_type": kind,
                             "well_type_label": "Intron controls" if kind == "CRISPR" else ""})
    splits = make_plate_splits({row["well_id"]: row for row in rows})
    vectors = []
    drug_vectors = [[0., 1., 0.], [1., 1., 0.], [-1., 1., 0.]]
    for row in rows:
        row["split"] = splits[row["experiment_name"], int(row["plate"])]
        if row["gene"]:
            vectors.append([1., 0., 0.] if row["gene"] == "G" else [-1., 0., 0.])
        elif row["treatment"].startswith("drug"):
            vectors.append(drug_vectors[int(row["treatment"][-1])])
        else:
            vectors.append([0., 0., 1.])
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path_a, path_b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    ids = [row["well_id"] for row in rows]
    pq.write_table(pa.table({"well_id": ids, "embedding": vectors}), path_a)
    wide = np.pad(np.array(vectors), ((0, 0), (0, 2)))
    pq.write_table(pa.table({"well_id": ids, **{f"feature_{i}": wide[:, i] for i in range(5)}}), path_b)
    output = tmp_path / "all_tasks"
    rules = json.loads(Path("configs/negative_controls.example.json").read_text(encoding="utf-8"))
    report = evaluate_embeddings(embeddings=[("encoder_A", path_a), ("encoder_B", path_b)],
                                 metadata_path=metadata, output_dir=output,
                                 tasks=["activity", "same-guide", "compound", "cross-guide"],
                                 control_rules=rules, n_permutations=29, n_null=20,
                                 n_bootstrap=10, max_controls=4)
    assert len(report["summary"]) == 8
    scores = {(row["model"], row["task"]): row for row in report["summary"]}
    for name in ("encoder_A", "encoder_B"):
        assert scores[name, "activity"]["map"] == 1
        assert scores[name, "activity"]["energy_distance"] > 0
        assert scores[name, "compound"]["percent_replicating"] == 100
        assert scores[name, "cross-guide"]["mrr"] == 1
        assert (output / name / "compound" / "replicates.csv").exists()
        assert (output / name / "activity" / "blocks.csv").exists()
    assert (output / "paired_differences.csv").exists()
