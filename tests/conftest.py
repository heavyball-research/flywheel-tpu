"""--shard=i/N keeps every N-th collected test from the i-th on, so N CI jobs that pass
i = 0 .. N-1 split the suite between them, each test running in exactly one job.

Off TPU, jax's caches are cleared after every test, which keeps the suite's memory flat."""

import gc

import jax
import pytest


def pytest_addoption(parser):
    parser.addoption("--shard", metavar="i/N", help="run only the tests at collected index i mod N")


def pytest_collection_modifyitems(config, items):
    shard = config.getoption("shard")
    if shard is None:
        return
    try:
        index, total = map(int, shard.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard takes i/N, not {shard!r}") from None
    if not 0 <= index < total:
        raise pytest.UsageError(f"--shard {shard}: need 0 <= i < N")
    # Note: collection order is fixed by the parametrize lists, so every job (and every
    # xdist worker in it) sees the same items and the shards partition them. The heavy
    # cases sit next to light ones in that order, so striding keeps them from piling up
    # in one shard.
    config.hook.pytest_deselected(items=[item for k, item in enumerate(items) if k % total != index])
    items[:] = items[index::total]


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    # Note: under the Pallas interpreter nearly every test compiles shapes of its own, and
    # jax keeps every executable for the life of the process. In CI each xdist worker grew
    # past 4 GB, and four of them filled a 16 GB runner until it was shut down. Clearing
    # after each test held a shard under 6.5 GB, for 17% more time. On TPU only one process
    # can hold the chip, so the suite runs in one process on a large host and keeps them.
    if jax.default_backend() != "tpu":
        jax.clear_caches()
        gc.collect()
