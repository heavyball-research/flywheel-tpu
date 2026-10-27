"""A persistent JAX compilation cache for the test suite.

At long sequence lengths most of a test's time is compilation, and much of it
repeats: two eager calls of one kernel compile it twice (pl.pallas_call builds
a fresh jit wrapper on every call), and sibling parametrizations compile the
same reference programs. The on-disk cache is keyed by the compiled program,
so every repeat, within a run or across runs, becomes a read.
JAX_COMPILATION_CACHE_DIR picks another directory.
"""

import os

import jax

if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "flywheel_tpu",
            "jax_compilation_cache",
        ),
    )
