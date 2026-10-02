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

"""Measured kernel configs per TPU generation and exact problem shape.

Keys are exact, not bucketed: q_block must divide the real seq_len and
head_fold the real num_heads, so a rounded-up entry could hand back a config
the shape cannot build. A missing key returns None and the caller uses its
analytic defaults. A kvcache| key holds the contiguous KV-cache decode
kernel's (block_kv, block_kv_compute, num_kv_stages); every other key holds a
dense fwd / bwd CONFIG_FIELDS tuple.
"""

import functools
import json
import pathlib
import re

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

TUNED_CONFIGS_PATH = pathlib.Path(__file__).with_name("tuned_configs.json")

# Note (david): field order of a dense fwd / bwd tuned_configs.json value,
# shared by every tuple get_tuned_config returns.
CONFIG_FIELDS = (
    "q_block",
    "kv_block",
    "q_cblock",
    "kv_cblock",
    "stages",
    "head_fold",
    "qkv_layout",
    "transposed_pv",
)

TunedConfig = tuple[int | str | bool, ...]
KVCacheConfig = tuple[int, int, int]

# Note (david): varlen runs a runtime multi-step schedule whose best fold
# differs from a dense single-block kernel at the same physical block span.
# Keys are (head_dim, block_span, causal): the runtime cu_seqlens stay out of
# the compile key and a span costs the same per step at any packed total, so
# this table cannot live in tuned_configs.json's exact-shape key space.
# Measured on TPU v6 lite with 16384 packed tokens, 16 heads, bf16 and uniform
# block-aligned sequences.
TUNED_VARLEN_FWD_HEAD_FOLD: dict[
    str, dict[tuple[int, int, bool], int]
] = {
    "TPU v6": {
        (128, 512, True): 8,
        (128, 512, False): 8,
        (128, 1024, True): 2,
        (128, 1024, False): 4,
    },
}


def get_device_variant_name(device_kind: str | None = None) -> str:
  """'TPU v6 lite' / 'TPU7x' -> 'TPU v6e' / 'TPU v7' (generation plus e/p).

  device_kind defaults to the chip JAX runs on.
  """
  if device_kind is None:
    device_kind = jax.devices()[0].device_kind
  kind_match = re.fullmatch(r"TPU\s*v?(\d+)(?:(e|p)|\s+(lite))?", device_kind)
  if device_kind == "TPU7x":
    return "TPU v7"
  elif kind_match is None:
    return device_kind
  elif kind_match.group(2):
    return f"TPU v{kind_match.group(1)}{kind_match.group(2)}"
  elif kind_match.group(3):
    return f"TPU v{kind_match.group(1)}e"
  else:
    return f"TPU v{kind_match.group(1)}"


def get_device_name(device_kind: str | None = None) -> str:
  """'TPU v6 lite' / 'TPU v5e' / ... -> 'TPU v6' (generation key, as JAX does)."""
  return re.sub(
      r"^(TPU v\d+)[ep]$", r"\1", get_device_variant_name(device_kind))


def get_varlen_head_fold(
    key: tuple[int, int, bool], num_heads: int, head_dim: int, head_dim_v: int,
) -> int:
  """Measured varlen forward head_fold for key, halved until it divides
  num_heads; 1 when untuned.

  Every entry was measured at head_dim_v == head_dim; the folded scratch
  scales with head_dim_v, so a wider value head is a different shape and gets
  no fold.
  """
  if head_dim_v == head_dim:
    device_table = TUNED_VARLEN_FWD_HEAD_FOLD.get(get_device_name(), {})
  else:
    device_table = {}
  head_fold = device_table.get(key, 1)
  while head_fold > 1 and num_heads % head_fold:
    head_fold //= 2
  return head_fold


def tuned_config_key(
    direction: str,
    *,
    token_major: bool,
    num_heads: int,
    batch: int | None,
    seq_len: int,
    head_dim: int,
    causal: bool,
    max_seqlen: int | None,
    return_lse: bool,
    num_kv_heads: int | None = None,
    heads_outer_batch: int | None = None,
) -> str:
  """The tuned_configs.json key for one kernel build.

  Every field is what the kernel sees: num_heads is the flat batch * heads
  count head-major and the per-row count token-major, batch is set only for
  3-D token-major refs, seq_len is the padded q axis (the whole packed buffer
  on varlen), and max_seqlen is the varlen schedule's bucket (None on dense).
  num_kv_heads (when it differs from num_heads) and heads_outer_batch extend
  the key only when present, so a GQA or heads-outer build never hits an entry
  measured on another head map.
  """
  base_key = (
      f"{direction}|{'tm' if token_major else 'hm'}|bfloat16"
      f"|h{num_heads}|b{'-' if batch is None else batch}|s{seq_len}|d{head_dim}"
      f"|c{int(bool(causal))}|m{'-' if max_seqlen is None else max_seqlen}"
      f"|l{int(bool(return_lse))}"
  )
  if num_kv_heads is not None and num_kv_heads != num_heads:
    kv_heads_suffix = f"|k{num_kv_heads}"
  else:
    kv_heads_suffix = ""
  if heads_outer_batch is not None:
    heads_outer_suffix = f"|o{heads_outer_batch}"
  else:
    heads_outer_suffix = ""
  return base_key + kv_heads_suffix + heads_outer_suffix


@functools.cache
def load_tuned_configs() -> dict[str, dict[str, list[int | str | bool]]]:
  return json.loads(TUNED_CONFIGS_PATH.read_text())


def get_tuned_config(
    direction: str, *, head_dim_v: int, **key_fields: int | bool | None
) -> TunedConfig | None:
  """Return the measured config tuple (CONFIG_FIELDS order), or None.

  head_dim_v is checked, not keyed: every entry was measured at
  head_dim_v == head_dim.
  """
  if head_dim_v != key_fields["head_dim"]:
    return None
  else:
    device_table = load_tuned_configs().get(get_device_name(), {})
    config = device_table.get(tuned_config_key(direction, **key_fields))
    return None if config is None else tuple(config)


def kvcache_config_key(
    *,
    dtype: DTypeLike,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
    batch: int,
    append: bool,
    sliding_window: int | None,
    return_lse: bool,
) -> str:
  """The tuned_configs.json key for one contiguous KV-cache decode build."""
  return (
      f"kvcache|{jnp.dtype(dtype).name}|h{num_query_heads}|k{num_kv_heads}"
      f"|b{batch}|s{capacity}|d{head_dim}|a{int(bool(append))}"
      f"|w{'-' if sliding_window is None else sliding_window}"
      f"|l{int(bool(return_lse))}"
  )


def get_tuned_kvcache_config(
    **key_fields: DTypeLike | int | bool | None,
) -> KVCacheConfig | None:
  device_table = load_tuned_configs().get(get_device_name(), {})
  config = device_table.get(kvcache_config_key(**key_fields))
  return None if config is None else tuple(config)
