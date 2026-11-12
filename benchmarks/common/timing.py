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

"""The timing protocol every benchmark shares: queued calls, one block per
round, median of the rounds."""

# jax loads only where a TPU run needs it, so the host-only paths never
# claim the TPU.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

WARMUP_CALLS = 3
REPEATS = 5
MIN_ROUND_S = 0.5
LONG_CALL_S = 10.0
LONG_REPEATS = 3


def time_call(step, state):
    """Median ms per call: queued calls, one block per round."""
    import jax

    start = time.perf_counter()
    out, state = jax.block_until_ready(step(state))
    compile_s = time.perf_counter() - start

    start = time.perf_counter()
    out, state = jax.block_until_ready(step(state))
    one_call_s = time.perf_counter() - start

    if one_call_s >= LONG_CALL_S:
        iters, repeats, warmup = 1, LONG_REPEATS, 1
    else:
        iters = max(1, math.ceil(MIN_ROUND_S / one_call_s))
        repeats, warmup = REPEATS, WARMUP_CALLS
        for _ in range(warmup - 1):
            out, state = step(state)
        jax.block_until_ready((out, state))

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            out, state = step(state)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) * 1e3 / iters)
    return {
        "compile_s": compile_s,
        "warmup_calls": warmup,
        "iters": iters,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
    }, out


def time_stateful(
    step: Callable[..., tuple[tuple[jax.Array, ...], jax.Array]],
    state: tuple[jax.Array, ...],
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[float, jax.Array]:
    """Median ms per call; step(*state) -> (new_state, out) chains donated
    buffers."""
    import jax

    out = None
    for _ in range(warmup):
        state, out = step(*state)
    jax.block_until_ready((state, out))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            state, out = step(*state)
        jax.block_until_ready((state, out))
        samples.append((time.perf_counter() - start) / iters * 1e3)
    return statistics.median(samples), out
