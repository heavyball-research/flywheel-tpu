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

"""Test setup: a compilation cache, --subset, --shard, cache clearing.

At long sequence lengths most of a test's time is compilation, and much of it
repeats: two eager calls of one kernel compile it twice (pl.pallas_call builds
a fresh jit wrapper on every call), and sibling parametrizations compile the
same reference programs. The on-disk cache is keyed by the compiled program,
so every repeat, within a run or across runs, becomes a read.
JAX_COMPILATION_CACHE_DIR picks another directory.

--subset runs every test except in the modules marked subset_sample(stride),
which keep one test in `stride`. The pick hashes the test function name and
its parametrization id, never the file path, so it survives moving or renaming
the test files.

--shard=i/N keeps every N-th collected test from the i-th on, so N CI jobs that
pass i = 0 .. N-1 split the suite between them, each test running in exactly
one job. With --subset, the shards split what the subset keeps.

Off TPU, jax's in-memory caches are cleared after every test, which keeps the
suite's memory flat. The on-disk cache is not cleared.
"""

import gc
import os
import zlib

import jax
import pytest

if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "flywheel_tpu",
            "jax_compilation_cache",
        ),
    )

SUBSET_MARKER = "subset_sample"


def pytest_addoption(parser):
    parser.addoption(
        "--subset",
        action="store_true",
        help=(
            "Run every test, but only a deterministic sample of the modules"
            f" marked {SUBSET_MARKER}(stride)."
        ),
    )
    parser.addoption(
        "--shard",
        metavar="i/N",
        help="run only the tests at collected index i mod N",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{SUBSET_MARKER}(stride): under --subset keep one test in `stride`,"
        " picked by a CRC32 of the test name and its parametrization id.",
    )


def _subset_stride(item):
    """The stride of the item's subset_sample marker, or None if unmarked."""
    marker = item.get_closest_marker(SUBSET_MARKER)
    if marker is None:
        return None
    if marker.kwargs or len(marker.args) != 1:
        raise ValueError(
            f"{SUBSET_MARKER} takes exactly one positional stride, got"
            f" args={marker.args} kwargs={marker.kwargs} on {item.nodeid}"
        )
    stride = marker.args[0]
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError(
            f"{SUBSET_MARKER} stride must be a positive int, got {stride!r}"
            f" on {item.nodeid}"
        )
    return stride


def _apply_subset(config, items):
    """Under --subset, deselect all but a stable sample of the marked modules.

    The sample keys on the test function name and the parametrization id
    only, never on the file path or node id, so moving or renaming a test file
    keeps the same selection.
    """
    if not config.getoption("subset"):
        return
    selected, deselected = [], []
    for item in items:
        stride = _subset_stride(item)
        if stride is None:
            selected.append(item)
            continue
        callspec = getattr(item, "callspec", None)
        param_id = "" if callspec is None else callspec.id
        key = f"{item.originalname}[{param_id}]"
        if zlib.crc32(key.encode()) % stride == 0:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _apply_shard(config, items):
    """Under --shard=i/N, keep every N-th item from the i-th on."""
    shard = config.getoption("shard")
    if shard is None:
        return
    try:
        index, total = map(int, shard.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard takes i/N, not {shard!r}") from None
    if not 0 <= index < total:
        raise pytest.UsageError(f"--shard {shard}: need 0 <= i < N")
    # Note: collection order is fixed by the parametrize lists, so every job
    # (and every xdist worker in it) sees the same items and the shards
    # partition them. The heavy cases sit next to light ones in that order, so
    # striding keeps them from piling up in one shard.
    config.hook.pytest_deselected(
        items=[item for k, item in enumerate(items) if k % total != index]
    )
    items[:] = items[index::total]


def pytest_collection_modifyitems(config, items):
    _apply_subset(config, items)
    _apply_shard(config, items)


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    # Note: under the Pallas interpreter nearly every test compiles shapes of
    # its own, and jax keeps every executable for the life of the process. In
    # CI each xdist worker grew past 4 GB, and four of them filled a 16 GB
    # runner until it was shut down. Clearing after each test held a shard
    # under 6.5 GB, for 17% more time. On TPU only one process can hold the
    # chip, so the suite runs in one process on a large host and keeps them.
    if jax.default_backend() != "tpu":
        jax.clear_caches()
        gc.collect()
