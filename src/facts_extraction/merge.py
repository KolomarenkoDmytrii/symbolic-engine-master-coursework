import json
from pathlib import Path
from typing import Iterable

import ijson


def merge_json_arrays(part_files: Iterable[Path], output_path: Path) -> None:
    """Merge JSON arrays from batch exports into one array, one record at a time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        output.write("[")
        first = True

        for part_path in part_files:
            with part_path.open("rb") as part:
                for record in ijson.items(part, "item"):
                    if not first:
                        output.write(",")
                    json.dump(record, output, ensure_ascii=False)
                    first = False

        output.write("]\n")
