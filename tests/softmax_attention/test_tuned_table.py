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

"""The tuned table as the source of truth for the analytic default configs.

Every tuned_configs.json entry names one kernel build. Each test resolves
that build's config through the code a real call runs (dispatch.get_kernel,
resolve_bwd_config, resolve_kvcache_config, default_mla_config), on the
entry's device, in two ways: with the table, and with the table lookup
switched off, which gives the analytic default. Nothing here runs a kernel,
so the tests need no TPU.

  test_every_key_names_a_known_build   no entry is left out of the tests
  test_entry_resolves                  with the table, each build looks up
                                       its own key and resolves a legal
                                       config that fits the device's VMEM
  test_default_agreement_with_table    per device and kernel family, the
                                       entries whose default is the tuned
                                       config, against AGREEMENT
  test_analytic_dense_fwd_config_of_entry
                                       without the table, each dense
                                       head-major forward build takes
                                       dispatch.default_dense_fwd_config's
                                       config, which builds, fits VMEM and
                                       has the entry's blocks, fold and layout
  test_default_dense_fwd_config_*      what that config is on chosen shapes

AGREEMENT records how far the defaults are from making the table
unnecessary: a family whose count equals its entries no longer needs them.
A change to a default moves the counts, and the test then asks for the
record to be updated, so the change shows in review. The comparison is of
configs, not of times: the table holds each search's winner without its
timings, so a default that differs from an entry may still be as fast.

What a key does not carry is assumed, and stated where it is used: a varlen
forward key has no q bucket, a kvcache key no page size or cache layout, an
mla key no page size, query tokens or lse flag.
"""

import collections
import re

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu.softmax_attention import dispatch
from flywheel_tpu.softmax_attention.kernels import decode as decode_kernel
from flywheel_tpu.softmax_attention.kernels import mla_decode as mla_kernel
from flywheel_tpu.softmax_attention.variant import Variant

# The VMEM limit a kernel is built against on each tuned device, as
# vmem_limit_bytes() returns it there (test_dispatch.py).
DEVICE_VMEM_LIMIT_BYTES = {"TPU v6": 100 * 2**20, "TPU v7": 64 * 2**20}
# The chip default_mla_config sizes its default blocks for.
MLA_CHIP = {"TPU v6": "TPU v6e", "TPU v7": "TPU v7"}
# MLA v2's serving page, in 32-bit cache words (512 bf16 tokens, 1024 fp8
# tokens), as benchmarks/softmax_attention/decode_microbench.py runs the MLA
# decode cells. An mla key does not carry its page size.
MLA_PAGE_WORDS = 256

DENSE_KEY = re.compile(
    r"(?P<direction>fwd|bwd)\|(?P<layout>hm|tm)\|bfloat16\|h(?P<h>\d+)"
    r"\|b(?P<b>-|\d+)\|s(?P<s>\d+)\|d(?P<d>\d+)\|c(?P<c>[01])"
    r"\|m(?P<m>-|\d+)\|l(?P<l>[01])(?:\|k(?P<k>\d+))?(?:\|o(?P<o>\d+))?"
)
KVCACHE_KEY = re.compile(
    r"kvcache\|(?P<dtype>\w+)\|h(?P<h>\d+)\|k(?P<k>\d+)\|b(?P<b>\d+)"
    r"\|s(?P<s>\d+)\|d(?P<d>\d+)\|a(?P<a>[01])\|w(?P<w>-|\d+)\|l(?P<l>[01])"
)
MLA_KEY = re.compile(
    r"mla\|(?P<dtype>\w+)\|r(?P<r>\d+)\|b(?P<b>\d+)\|kv(?P<kv>\d+)"
)
KVCACHE_FIELDS = (
    "block_kv",
    "block_kv_compute",
    "num_kv_stages",
    "items_per_step",
)
MLA_FIELDS = ("block_kv", "items_per_step")
# A family's config fields that no build of it reads, left out of the
# comparison. The backward never reads block_q_compute (config.BlockSizes:
# "read only by the forward").
IGNORED_FIELDS = {"bwd": ("q_cblock",)}

# Entries whose analytic default is the tuned config, per device and family.
# A count equal to the family's entries means the family no longer needs the
# table. Raise a count when a default improves; a count that falls needs a
# reason in the change that lowers it.
AGREEMENT = {
    ("TPU v6", "bwd"): 2,  # of 36
    ("TPU v6", "fwd dense"): 60,  # of 134
    ("TPU v6", "kvcache"): 1,  # of 50
    ("TPU v6", "mla"): 93,  # of 333
    ("TPU v7", "fwd dense"): 18,  # of 37
    ("TPU v7", "fwd varlen"): 43,  # of 66
    ("TPU v7", "kvcache"): 2,  # of 123
    ("TPU v7", "mla"): 120,  # of 333
}


def parse_key(key):
    """(family, key fields) of a table key, or None for an unknown key."""
    fields = DENSE_KEY.fullmatch(key)
    if fields is not None:
        if fields["direction"] == "bwd":
            return "bwd", fields
        elif fields["m"] == "-":
            return "fwd dense", fields
        else:
            return "fwd varlen", fields
    fields = KVCACHE_KEY.fullmatch(key)
    if fields is not None:
        return "kvcache", fields
    fields = MLA_KEY.fullmatch(key)
    if fields is not None:
        return "mla", fields
    return None


def table_entries():
    """(device name, key, tuned config) of every table entry."""
    for device_name, table in dispatch.load_table(
        dispatch.TUNED_CONFIGS_PATH
    ).items():
        for key, config in table.items():
            yield device_name, key, config


def config_fields(family):
    if family == "kvcache":
        return KVCACHE_FIELDS
    elif family == "mla":
        return MLA_FIELDS
    else:
        return dispatch.CONFIG_FIELDS


def dense_variant(fields):
    return Variant(
        softmax_scale=int(fields["d"]) ** -0.5,
        causal=fields["c"] == "1",
        return_lse=fields["l"] == "1",
    )


def resolve_fwd(fields, monkeypatch, build_kernel=False):
    """The config dispatch.get_kernel hands the kernel builder for a forward
    key, in CONFIG_FIELDS order; with build_kernel, the builder runs on it
    and its kernel is returned instead.

    A varlen key carries the kv bucket but not the q bucket, which is taken
    as unset, as a varlen call without chunked prefill leaves it.
    """

    def record(*_, block_sizes, head_fold, transposed_pv, **__):
        return (
            block_sizes.block_q,
            block_sizes.block_kv,
            block_sizes.block_q_compute,
            block_sizes.block_kv_compute,
            block_sizes.num_stages,
            head_fold,
            block_sizes.qkv_layout.name.lower(),
            transposed_pv,
        )

    num_heads, seq_len = int(fields["h"]), int(fields["s"])
    with monkeypatch.context() as patch:
        if not build_kernel:
            patch.setattr(dispatch, "make_flash_attn_mha", record)
        # get_kernel caches on its arguments, which do not hold the table.
        return dispatch.get_kernel.__wrapped__(
            num_heads=num_heads,
            num_kv_heads=int(fields["k"] or num_heads),
            seqlen_q=seq_len,
            seqlen_kv=seq_len,
            variant=dense_variant(fields),
            interpret=True,
            head_dim=int(fields["d"]),
            head_dim_v=int(fields["d"]),
            varlen_max_seqlen_kv=(
                None if fields["m"] == "-" else int(fields["m"])
            ),
            token_major=fields["layout"] == "tm",
            kernel_batch=None if fields["b"] == "-" else int(fields["b"]),
            heads_outer_batch=(
                None if fields["o"] is None else int(fields["o"])
            ),
        )


def resolve_bwd(fields):
    """The config dispatch.resolve_bwd_config picks for a backward key, in
    CONFIG_FIELDS order. Every backward key is head-major MHA."""
    assert fields["layout"] == "hm" and fields["k"] is None, fields.string
    # resolve_bwd_config reads its operands' shapes only.
    operand = jax.ShapeDtypeStruct(
        (int(fields["h"]), int(fields["s"]), int(fields["d"])), jnp.bfloat16
    )
    block_sizes, head_fold = dispatch.resolve_bwd_config(
        operand,
        operand,
        operand,
        variant=dense_variant(fields),
        token_major=False,
        head_dim=None,
    )
    return (
        block_sizes.block_q,
        block_sizes.block_kv,
        block_sizes.block_q_compute,
        block_sizes.block_kv_compute,
        block_sizes.num_stages,
        head_fold,
        block_sizes.qkv_layout.name.lower(),
        False,
    )


def resolve_kvcache(fields):
    """The validated config dispatch.resolve_kvcache_config picks for a
    kvcache key, in KVCACHE_FIELDS order.

    A key carries neither the cache layout nor the page size. The entries are
    one-item configs, which a merged-cache decode of two or more rows skips
    for its packed default, so the build is taken as the contiguous K/V pair
    of the key's capacity, where an entry always applies.
    """
    window = -1 if fields["w"] == "-" else int(fields["w"])
    return dispatch.resolve_kvcache_config(
        page_size=None,
        dtype=jnp.dtype(fields["dtype"]),
        num_query_heads=int(fields["h"]),
        num_kv_heads=int(fields["k"]),
        head_dim=int(fields["d"]),
        capacity=int(fields["s"]),
        batch=int(fields["b"]),
        append=fields["a"] == "1",
        variant=Variant(
            softmax_scale=1.0,
            window_size=(window, -1),
            has_kv_scales=True,
            return_lse=fields["l"] == "1",
        ),
        merged_cache=False,
        query_tokens=1,
    )


def resolve_mla(fields):
    """The fitted config mla_decode.default_mla_config picks for an mla key,
    in MLA_FIELDS order.

    A key carries its cache dtype, query rows and its batch and max_seqlen_k
    buckets. The build is taken at the buckets' values, one query token a
    request, no lse, and MLA_PAGE_WORDS pages that hold max_seqlen_k.
    """
    cache_dtype = jnp.dtype(fields["dtype"])
    page_size = MLA_PAGE_WORDS * mla_kernel.cache_packing(cache_dtype)
    max_seqlen_k = int(fields["kv"])
    return tuple(
        mla_kernel.default_mla_config(
            batch=int(fields["b"]),
            page_size=page_size,
            pages_per_seq=-(-max_seqlen_k // page_size),
            num_query_rows=int(fields["r"]),
            query_tokens=1,
            cache_dtype=cache_dtype,
            return_lse=False,
            max_seqlen_k=max_seqlen_k,
        )
    )


def resolve(family, fields, monkeypatch):
    """The config the build a key names resolves to, as dispatch sees the
    device and the table when it is called."""
    if family == "bwd":
        return resolve_bwd(fields)
    elif family == "kvcache":
        return resolve_kvcache(fields)
    elif family == "mla":
        return resolve_mla(fields)
    else:
        return resolve_fwd(fields, monkeypatch)


@pytest.fixture(name="run_on")
def fixture_run_on(monkeypatch):
    """run_on(device_name) makes the config resolution see that device: its
    table section, its VMEM limit and its MLA default blocks."""

    def pretend(device_name):
        limit = DEVICE_VMEM_LIMIT_BYTES[device_name]
        monkeypatch.setattr(dispatch, "get_device_name", lambda: device_name)
        for module in (dispatch, decode_kernel, mla_kernel):
            monkeypatch.setattr(module, "vmem_limit_bytes", lambda: limit)
        monkeypatch.setattr(mla_kernel, "INTERPRET_CHIP", MLA_CHIP[device_name])

    return pretend


def test_every_key_names_a_known_build():
    unknown = [
        (device_name, key)
        for device_name, key, _ in table_entries()
        if parse_key(key) is None or device_name not in DEVICE_VMEM_LIMIT_BYTES
    ]
    assert not unknown


@pytest.mark.parametrize(
    "device_name, key, tuned",
    [
        pytest.param(device_name, key, tuned, id=f"{device_name}|{key}")
        for device_name, key, tuned in table_entries()
        if parse_key(key) is not None
    ],
)
def test_entry_resolves(device_name, key, tuned, monkeypatch, run_on):
    """With the table, the build a key names looks up that key, and its
    config passes the checks of the code that resolves it: the kernel
    builder's for a forward entry, the decode kernel's validator and VMEM
    model for a kvcache entry, the MLA kernel's VMEM budget for an mla
    entry."""
    run_on(device_name)
    family, fields = parse_key(key)
    lookup, hits = dispatch.lookup, []

    def spy(table, lookup_key):
        hits.append(lookup_key)
        return lookup(table, lookup_key)

    monkeypatch.setattr(dispatch, "lookup", spy)
    if family in ("fwd dense", "fwd varlen"):
        # The builder checks the config and returns the kernel untraced.
        assert callable(resolve_fwd(fields, monkeypatch, build_kernel=True))
    config = resolve(family, fields, monkeypatch)
    assert key in hits, f"the build looked up {sorted(set(hits))}"
    if family == "bwd":
        tuned_fields = dict(zip(dispatch.CONFIG_FIELDS, tuned))
        seq_len = int(fields["s"])
        # resolve_bwd_config's own statement of what an entry must satisfy.
        assert seq_len % tuned_fields["q_block"] == 0
        assert seq_len % tuned_fields["kv_block"] == 0
        assert tuned_fields["kv_block"] % tuned_fields["kv_cblock"] == 0
        assert tuned_fields["head_fold"] == 1 or (
            tuned_fields["q_block"] == tuned_fields["kv_block"] == seq_len
        )
    elif family == "kvcache":
        # resolve_kvcache_config validated it; it is the entry as stored.
        assert config[: len(tuned)] == tuple(tuned)
    elif family == "mla":
        # The kernel fits the entry's target block and items cap to the
        # pages and the VMEM budget; it never exceeds the cap.
        assert config[1] <= tuned[1]


def agreement_groups():
    groups = collections.defaultdict(list)
    for device_name, key, _ in table_entries():
        parsed = parse_key(key)
        if parsed is not None:
            groups[(device_name, parsed[0])].append((key, parsed[1]))
    return groups


def test_default_agreement_with_table(monkeypatch, run_on):
    """Per device and family, the entries whose build resolves to the same
    config with the table and without it, against AGREEMENT."""
    agreement = {}
    report = []
    for (device_name, family), entries in sorted(agreement_groups().items()):
        run_on(device_name)
        with_table = [
            resolve(family, fields, monkeypatch) for _, fields in entries
        ]
        with monkeypatch.context() as patch:
            patch.setattr(dispatch, "lookup", lambda table, key: None)
            without_table = [
                resolve(family, fields, monkeypatch) for _, fields in entries
            ]
        names = config_fields(family)
        compared = [
            index
            for index, name in enumerate(names)
            if name not in IGNORED_FIELDS.get(family, ())
        ]
        differing = collections.Counter()
        agreed = 0
        for tuned, default in zip(with_table, without_table):
            fields_off = [
                names[index]
                for index in compared
                if tuned[index] != default[index]
            ]
            differing.update(fields_off)
            agreed += not fields_off
        agreement[(device_name, family)] = agreed
        report.append(
            f"    {(device_name, family)!r}: {agreed},"
            f"  # of {len(entries)}; differing: {dict(differing)}"
        )
    record = "AGREEMENT = {\n" + "\n".join(report) + "\n}"
    print(record)
    assert agreement == AGREEMENT, (
        "the defaults' agreement with the tuned table changed; if the"
        f" change is meant, record it:\n{record}"
    )


def dense_hm_fwd_entries():
    """The dense head-major forward entries, the builds
    dispatch.default_dense_fwd_config serves when the table has no entry."""
    for device_name, key, tuned in table_entries():
        parsed = parse_key(key)
        if (
            parsed is not None
            and parsed[0] == "fwd dense"
            and parsed[1]["layout"] == "hm"
        ):
            yield pytest.param(
                device_name, key, tuned, id=f"{device_name}|{key}"
            )


# What the analytic config has in common with every dense head-major forward
# entry. Its compute tiles and stages differ from some entries'; whether those
# are as fast is a timing question this file cannot answer.
ANALYTIC_FWD_STRUCTURE = (
    "q_block",
    "kv_block",
    "head_fold",
    "qkv_layout",
    "transposed_pv",
)


@pytest.mark.parametrize(
    "device_name, key, tuned", list(dense_hm_fwd_entries())
)
def test_analytic_dense_fwd_config_of_entry(
    device_name, key, tuned, monkeypatch, run_on
):
    """Without the table, the build a dense head-major forward key names
    takes dispatch.default_dense_fwd_config's config; the kernel builder
    accepts it, a fold's staging fits the device's VMEM budget, and it has
    the entry's blocks, head fold and layout."""
    run_on(device_name)
    _, fields = parse_key(key)
    monkeypatch.setattr(dispatch, "lookup", lambda table, lookup_key: None)
    # The builder checks the config and returns the kernel untraced.
    assert callable(resolve_fwd(fields, monkeypatch, build_kernel=True))
    resolved = resolve_fwd(fields, monkeypatch)
    num_heads, head_dim = int(fields["h"]), int(fields["d"])
    analytic = dispatch.default_dense_fwd_config(
        num_heads=num_heads,
        num_kv_heads=int(fields["k"] or num_heads),
        seq_len=int(fields["s"]),
        head_dim=head_dim,
        head_dim_v=head_dim,
        causal=fields["c"] == "1",
        return_lse=fields["l"] == "1",
        heads_outer_batch=None if fields["o"] is None else int(fields["o"]),
        device_name=device_name,
        vmem_limit=DEVICE_VMEM_LIMIT_BYTES[device_name],
    )
    assert resolved == analytic
    config = dict(zip(dispatch.CONFIG_FIELDS, resolved))
    tuned_fields = dict(zip(dispatch.CONFIG_FIELDS, tuned))
    assert [config[name] for name in ANALYTIC_FWD_STRUCTURE] == [
        tuned_fields[name] for name in ANALYTIC_FWD_STRUCTURE
    ]
    if config["head_fold"] > 1:
        staging = dispatch.dense_fold_vmem_bytes(
            config["head_fold"], config["q_block"], head_dim
        )
        assert (
            staging
            <= dispatch.DENSE_FOLD_VMEM_FRACTION
            * DEVICE_VMEM_LIMIT_BYTES[device_name]
        )


def analytic_fwd_config(
    device_name,
    num_heads,
    num_kv_heads,
    seq_len,
    head_dim,
    causal,
    return_lse=False,
    heads_outer_batch=None,
    head_dim_v=None,
):
    return dispatch.default_dense_fwd_config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        head_dim_v=head_dim if head_dim_v is None else head_dim_v,
        causal=causal,
        return_lse=return_lse,
        heads_outer_batch=heads_outer_batch,
        device_name=device_name,
        vmem_limit=DEVICE_VMEM_LIMIT_BYTES.get(device_name, 100 * 2**20),
    )


V6, V7 = "TPU v6", "TPU v7"


def square_config(block, q_tile, kv_tile, head_fold=1, seq_minor=False):
    """A CONFIG_FIELDS tuple of equal q and kv blocks at two stages."""
    layout = "seq_minor" if seq_minor else "head_dim_minor"
    return (block, block, q_tile, kv_tile, 2, head_fold, layout, seq_minor)


@pytest.mark.parametrize(
    "device_name, num_heads, num_kv_heads, seq_len, head_dim, causal, config",
    [
        # The MXU-wide kv tile, unfolded: GQA, or MHA past 1024 tokens.
        (V7, 16, 4, 128, 128, True, square_config(128, 64, 128)),
        (V7, 16, 4, 8192, 128, True, square_config(2048, 256, 256)),
        (V7, 16, 4, 8192, 128, False, square_config(2048, 256, 256)),
        (V7, 32, 32, 131072, 256, True, square_config(2048, 256, 256)),
        (V7, 64, 64, 2048, 128, True, square_config(2048, 256, 256)),
        (V6, 32, 32, 8192, 128, False, square_config(2048, 256, 256)),
        # v6e at heads of 256: the 512-key tile under a full mask or from
        # four blocks a row, the MXU-wide one for a shorter causal row.
        (V6, 16, 4, 8192, 256, True, square_config(2048, 256, 512)),
        (V6, 16, 4, 2048, 256, False, square_config(2048, 256, 512)),
        (V6, 16, 4, 4096, 256, False, square_config(2048, 256, 512)),
        (V6, 16, 4, 4096, 256, True, square_config(2048, 256, 256)),
        (V6, 16, 4, 2048, 256, True, square_config(2048, 256, 256)),
        (V6, 16, 4, 1024, 256, True, square_config(1024, 256, 256)),
        # Folds: 16 heads of 512 tokens and 4 of 1024, at the 512-key tile.
        (V7, 512, 512, 512, 128, True, square_config(512, 256, 512, 16)),
        (V6, 512, 512, 512, 128, False, square_config(512, 256, 512, 16)),
        (V7, 256, 256, 1024, 128, True, square_config(1024, 256, 512, 4)),
        (V6, 128, 128, 1024, 256, True, square_config(1024, 256, 512, 4)),
        # 16 heads of 256 over 512 tokens pass the VMEM budget on both chips.
        (V7, 256, 256, 512, 256, True, square_config(512, 256, 512, 8)),
        (V6, 256, 256, 512, 256, True, square_config(512, 256, 512, 8)),
        # The fold is a power of two that divides the heads.
        (V7, 24, 24, 512, 128, True, square_config(512, 256, 512, 8)),
        (V7, 3, 3, 512, 128, True, square_config(512, 256, 256)),
        # No fold under 512 tokens.
        (V7, 512, 512, 256, 128, True, square_config(256, 128, 256)),
        # Heads of 64: SEQ_MINOR with the transposed PV from 512 tokens a
        # block, with 512-row q tiles where the block holds two.
        (V6, 16, 4, 8192, 64, True, square_config(2048, 512, 512, 1, True)),
        (V7, 16, 4, 8192, 64, False, square_config(2048, 512, 512, 1, True)),
        (V6, 512, 512, 1024, 64, True, square_config(1024, 512, 512, 4, True)),
        (V6, 1024, 1024, 512, 64, True, square_config(512, 256, 512, 16, True)),
        (V7, 16, 4, 256, 64, True, square_config(256, 128, 256)),
    ],
)
def test_default_dense_fwd_config_picks(
    device_name, num_heads, num_kv_heads, seq_len, head_dim, causal, config
):
    assert (
        analytic_fwd_config(
            device_name, num_heads, num_kv_heads, seq_len, head_dim, causal
        )
        == config
    )


def test_default_dense_fwd_config_keeps_to_what_the_kernel_builds():
    # heads_outer_batch runs unfolded only.
    assert analytic_fwd_config(
        V7, 512, 512, 512, 128, True, heads_outer_batch=4
    ) == square_config(512, 256, 256)
    # The transposed PV does not return lse.
    assert analytic_fwd_config(
        V6, 16, 4, 8192, 64, True, return_lse=True
    ) == square_config(2048, 256, 256)


@pytest.mark.parametrize("device_name", ["cpu", "TPU v5", "TPU v5e", "TPU v4"])
def test_default_dense_fwd_config_is_for_measured_devices(device_name):
    assert analytic_fwd_config(device_name, 16, 4, 8192, 128, True) is None


def test_default_dense_fwd_config_needs_equal_head_widths():
    assert (
        analytic_fwd_config(V7, 16, 4, 8192, 128, True, head_dim_v=256) is None
    )
