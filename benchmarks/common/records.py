"""JSONL result records: append one, and look up whether a cell is done."""

from __future__ import annotations

import json


def write_record(path, record):
    """Append one benchmark result when an output path is supplied."""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def recorded(path, key):
    """Whether the JSONL at path already holds an ok record of this cell."""
    if path is None or not path.exists():
        return False
    with path.open() as handle:
        for line in handle:
            # Note (david): a run killed mid-write leaves a partial last line.
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("key") == key and record.get("status") == "ok":
                return True
    return False
