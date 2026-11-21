# MLA decode block heuristic

`default_mla_config` chooses `(block_kv, items_per_step)` for MLA decode. On
TPU v6e with a bfloat16 cache it now minimises a calibrated cost model over
every block the kernel accepts, instead of following a fixed `max_seqlen`
rule. This note records the idea and the design.

## Why the old rule was not enough

The legacy rule keys off `max_seqlen` alone and cannot trade block size
against a workload's length distribution. An earlier calibrated version that
used a coarse candidate set and a short length prior still reached mean regret
1.51% on page 128 and 2.64% on page 1024 over held-out length patterns. Two
effects drive the gap: larger blocks amortise the per-step cost, and a
partially filled item still pays a floor fraction of a full block.

## Candidate set

A candidate is every block the kernel accepts at the call's page size, not a
power-of-two subset:

- `block_kv` is a multiple of 128 (`NUM_LANES`) and page-compatible:
  `block_kv % page == 0` or `page % block_kv == 0`;
- `block_kv <= 2048`;
- `items_per_step` ranges over `1..min(8, 16384 // block_kv)`.

When `max_items >= 4`, items 1 and 2 are pruned: they were measured 56% and
15% slower than items 8, while items 3-7 are within 2.6%. Keeping them changes
the page-1024 mean regret by only 0.08 percentage points.

## Cost model

For a candidate (B, ips) on a call's page, the expected time is

```
C(B, ips, page) = Σᵢ wᵢ · [ e(B, page) · gᵢ(B, page) · stepsᵢ · ips · B
                            + r_item · stepsᵢ · ips
                            + r_step(page) · stepsᵢ ]
```

- i indexes the 12 length-prior components: 3 length scales x 4 shapes; the
  scales are 512 / 2048 / 8192 tokens with weights 0.40 / 0.45 / 0.15;
- stepsᵢ = max(1, ceil(batch · Eᵢ[ceil(w / B)] / ips)), where w is a request's
  length;
- e(B, page) is the per-column efficiency of a full block;
- r_item and r_step(page) are fixed per-item and per-step costs (calibration
  fixes r_item = 0);
- gᵢ(B, page) is the fraction of a full block an item actually pays.

With x = an item's valid length / B, the floor model is

```
g(x, f) = f + (1 − f) · clamp(2x − 1, 0, 1)
```

where f = f(B, page) is the floor: an item pays f of a full block even when it
holds few tokens, and the remaining work ramps in over the upper half of the
block. f = 1 is the old full-block model.

The heuristic enumerates the candidates, drops those over the VMEM budget
(0.75 x 128 MiB) or the SMEM budget, picks the cheapest, and breaks ties
toward the longer block. It runs once per JIT trace; only the chosen
configuration is compiled.

## Calibration

Constants are per (chip, cache dtype). TPU v6e / bfloat16 is calibrated in
`CALIBRATION` from measurements at page 128 (the base table), page 512 and
page 1024 (exact per-page overrides for efficiency, floor and step cost; no
interpolation). The measured step costs are 153.6 / 249.6 / 257.4 columns.
Using the page-128 step and floor values for page 512 raises its validation
regret from 0.46% to 2.66%, so the three tables are overridden together.

A chip or dtype without an entry falls back to `tuned_configs.json`, then to
the legacy rule (with a one-time warning). With no constants at all the
heuristic stays dormant and the full lattice is compared with placeholder
efficiencies.

## Validation

Held-out length patterns (uniform, spiky, longtail, pareto; batch 64, mean
length 8192) are timed on every legal (block, items) cell, so the oracle is the
measured best over the lattice, including the pruned items.

| page | mean regret | max regret |
|---|---|---|
| 128 | 0.92% | 1.58% |
| 512 | 0.46% | 0.56% |
| 1024 | 0.42% | 0.61% |

The length prior was checked on independent workloads: 9 groups of 64k/128k
context and prior-sampled draws gave mean 0.93% / max 2.77% with the current
512/2048/8192 prior, versus 1.36% / 2.08% with the previous 256/1024/4096
prior. One group was worse with the long prior, so the choice is not uniformly
better.

Compile time is one configuration per trace. At batch 64 x 8192 tokens on
v6e, the heuristic's 384/8 and 512/8 choices compile in 3.0-3.4 s end to end,
versus 7.9-10.3 s for the legacy 4096/4 choice; the enumeration itself adds
0.1-0.3 s of Python to the trace.

## Reproduce

Calibrate and validate one page (repeat with `--page-size` 512 and 128):

```
PYTHONPATH=. python benchmarks/softmax_attention/calibrate_mla_decode_config.py \
    --out benchmarks/softmax_attention/results/mla_calibration_dense_p1024 \
    --page-size 1024 --floor-scan --validate
```

Refit from saved measurements without the TPU:

```
PYTHONPATH=. python benchmarks/softmax_attention/refit_mla_floor.py
```
