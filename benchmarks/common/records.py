# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
