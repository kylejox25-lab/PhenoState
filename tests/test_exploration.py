import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phenostate.data.exploration import read_selected_wells


def shard(path, keys):
    pq.write_table(pa.table({"__key__": keys, "jp2": [{"bytes": key.encode()} for key in keys]}),
                   path, row_group_size=2)
    return path


def test_selected_well_channels_span_shards_and_row_groups(tmp_path):
    keys = [f"gene-001/Plate2/A01_s1_{channel}" for channel in range(1, 7)]
    paths = [shard(tmp_path / "a.parquet", keys[:3]),
             shard(tmp_path / "b.parquet", keys[3:] + ["irrelevant/Plate1/A02_s1_1"])]
    wells = read_selected_wells(paths, ["gene-001_2_A01"])
    assert list(wells) == ["gene-001_2_A01"]
    assert wells["gene-001_2_A01"].channels == tuple(key.encode() for key in keys)


def test_missing_and_duplicate_channel_images_are_rejected(tmp_path):
    key = "gene-001/Plate2/A01_s1_1"
    path = shard(tmp_path / "images.parquet", [key])
    with pytest.raises(ValueError, match="Incomplete"):
        read_selected_wells([path], ["gene-001_2_A01"])
    path = shard(path, [key, key])
    with pytest.raises(ValueError, match="duplicate"):
        read_selected_wells([path], ["gene-001_2_A01"])
