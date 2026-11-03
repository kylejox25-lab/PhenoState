"""Read selected original wells, scanning keys before loading image row groups."""

from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .source import CHANNELS, Well, parse_image_key


def read_selected_wells(paths: Iterable[Path], well_ids: Iterable[str]) -> dict[str, Well]:
    """Read only row groups containing requested IDs; never decode other wells.

    No index or cache is written to the source dataset. Each selected well must
    have exactly one image for each of its six channels, including at boundaries
    between row groups or files.
    """
    wanted = list(dict.fromkeys(well_ids))
    if not wanted:
        raise ValueError("Select at least one well")
    keys = []
    for well_id in wanted:
        experiment, plate, address = well_id.rsplit("_", 2)
        keys.extend(f"{experiment}/Plate{int(plate)}/{address}_s1_{channel}" for channel in CHANNELS)
    key_values = pa.array(keys)
    images = {well_id: {} for well_id in wanted}
    locations = {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            key_table = parquet.read_row_group(group, columns=["__key__"])
            selected = pc.is_in(key_table["__key__"], value_set=key_values)
            if not pc.any(selected).as_py():
                continue
            table = parquet.read_row_group(group, columns=["__key__", "jp2"]).filter(selected)
            for sample in table.to_pylist():
                key = parse_image_key(sample["__key__"])
                raw = (sample["jp2"] or {}).get("bytes")
                if not raw or key.channel in images[key.well_id]:
                    raise ValueError(f"Missing or duplicate channel image: {sample['__key__']}")
                images[key.well_id][key.channel] = raw
                locations[key.well_id] = key
    result = {}
    for well_id, channels in images.items():
        if set(channels) != set(CHANNELS):
            raise ValueError(f"Incomplete or missing well {well_id}: found channels {sorted(channels)}")
        result[well_id] = Well(locations[well_id], tuple(channels[i] for i in CHANNELS))
    return result
