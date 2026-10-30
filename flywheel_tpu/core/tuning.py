# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tuning tables: measured kernel configs per TPU generation.

A table is a JSON file that maps a device name (get_device_name: 'TPU v6',
'TPU v7', ...) to the configs measured on that chip, each under the exact key
of the kernel build it was measured for. The family that owns a table builds
its keys and interprets its configs; this module only loads a table, looks a
key up for the current chip and warns when a TPU build falls back to its
default config.
"""

import functools
import json
import pathlib
import warnings

from .tpu import get_device_name

TableConfig = list[int | str | bool]


@functools.cache
def load_table(table: pathlib.Path) -> dict[str, dict[str, TableConfig]]:
    return json.loads(table.read_text(encoding="utf-8"))


@functools.cache
def _warn_untuned_once(
    table: pathlib.Path, device_name: str, key: str, reason: str
) -> None:
    detail = f" ({reason})" if reason else ""
    warnings.warn(
        f"flywheel_tpu: no usable tuned config for {key} on {device_name} in"
        f" {table.name}{detail}; falling back to the default config."
    )


def warn_untuned(table: pathlib.Path, key: str, reason: str = "") -> None:
    """Warn, once per table, device, key and reason, that a kernel build on a
    TPU falls back to its default config. Off TPU there are no tables to
    miss."""
    device_name = get_device_name()
    if device_name.startswith("TPU"):
        _warn_untuned_once(table, device_name, key, reason)


def lookup(table: pathlib.Path, key: str) -> TableConfig | None:
    """The current device's entry for key in table, warning when there is
    none."""
    config = load_table(table).get(get_device_name(), {}).get(key)
    if config is None:
        warn_untuned(table, key)
    return config
