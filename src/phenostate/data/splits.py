"""Save the project's deterministic plate split without converting images."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .source import load_metadata, make_plate_splits


def export_plate_splits(metadata_path: Path, output: Path, seed: int = 2026) -> dict:
    metadata = load_metadata(metadata_path)
    assignments = make_plate_splits(metadata, seed=seed)
    records = [{"experiment_name": experiment, "plate": plate, "split": split}
               for (experiment, plate), split in sorted(assignments.items())]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    counts = Counter(assignments[row["experiment_name"], int(row["plate"])] for row in metadata.values())
    return {"output": str(output.resolve()), "seed": seed, "n_plates": len(records), "well_counts": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New plate_splits.json; existing files are preserved")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(export_plate_splits(args.metadata, args.output, args.seed)))


if __name__ == "__main__":
    main()
