# MLA prefill block heuristic

`default_mla_prefill_config` chooses `(block_q, block_kv, row_groups)` for one
prefill call with the calibrated cost model below. TPU v6e with a bfloat16
cache is enabled; a chip or dtype without an entry falls back to the legacy
`max_seqlen` rule (one warning; no warning while `CALIBRATION` is empty).

## Terms

- `block_q`: query tokens covered by one q block; the kernel loops over q
  blocks.
- `block_kv`: key/value columns covered by one KV item; the kernel loops over
  items inside a q block.
- `row_groups`: how many groups the `block_q × H` rows of a q block are split
  into, with H = 32 heads.
- `group_rows = (block_q / row_groups) × H`: rows per group. Different
  `(block_q, row_groups)` pairs can reach the same `group_rows`; the model uses
  only its measured efficiency factor.
- VMEM: on-core vector memory. The KV ring, query staging, state and score
  buffers must fit 0.75 of the limit, or the configuration is not a candidate.
- Regret: chosen configuration's measured time / the workload's fastest
  measured configuration - 1. The acceptance criterion is at most 2% per
  validation workload.

## Cost model

The call's static shape gives batch B, total query tokens T, longest chunk
Qmax, longest context Kmax, page width and capacity. The length prior expands
it into weighted (q, n) samples; for each candidate the expected time is

```
C(config) = p(block_q) · h(group_rows) · e(block_kv) · W
          + c_blocks · blocks + c_rows · rows
          + c_qcopy · q_tokens + c_item_rows · item_rows
```

- W = Σ block_q · H · cols, where cols is the number of key columns the kernel
  actually computes;
- blocks = Σ ceil(q / block_q), the total number of q blocks in the call;
- rows = Σ block_q, q_tokens = Σ q_count · H;
- item_rows = Σ m · block_q · H, where m is the number of items in a q block;
- p, h and e are measured efficiency factors, with p(128) = 1 and
  h(256) = 1; e is the cost of one score element (one query × one head × one
  key column), and W already multiplies in block_q, H and the column count.

The factors are separate because v6e lets them be measured separately:
changing only the width or only the row grouping changes time even when the
column count is identical.

For item j of a q block, with F = n - q + j · block_q,
E = F + q_count and m = max(ceil(E / block_kv), 1):

- the first m - 1 items are full blocks of block_kv columns;
- the last item takes the live-tile path only when block_kv is a multiple of
  256 and greater than 256; it computes 256 · ceil(span / 256) columns, where
  span = E - (m - 1) · block_kv. Otherwise the last item also pays a full
  block.

When block_kv is a multiple of 256 this rule is equivalent to computing
256 · ceil(E / 256) columns for the whole q block. Between the large
candidates, changing block_kv therefore does not change W: it changes the item
count and the per-item fixed cost, which is how the model separates 512, 768
and 1024.

## Length prior

The prior reuses MLA decode's `WORKLOAD_PRIOR`: 12 components = 3 length
scales x 4 shapes (uniform, spiky, longtail, pareto), each sampled at 8
equal-mass quantiles. For one call a single scale s is solved with 80
bisection steps so that

```
E[min(s · w, Qmax)] = T / B
```

then q = min(s · w, Qmax) and n = min(s · w, Kmax, capacity), with n >= q.
The sample weights sum to 1 and the expansion is deterministic. When the
average chunk is shorter than Qmax this mixes full-prompt requests with
requests that have only part of a chunk; when every request has a full chunk,
context lengths spread up to Kmax.

The prior is a deployment assumption, not a fitted parameter. Replace
`WORKLOAD_PRIOR` with the deployment's length statistics rather than tuning
weights on a validation set.

## Candidate set and pruning

- `block_kv` must be page-aligned: `block_kv % page == 0` or
  `page % block_kv == 0`, and at most
  `min(2048, next_pow2(min(Kmax, capacity)))`. Page width only changes which
  blocks are legal; the same configuration differs by at most 0.7% between
  page 128 and page 512, and page 256/1024 spot checks agree with page 512.
- Only blocks with a measured entry (128..2048, every 128) are candidates. An
  unmeasured block is not priced with a placeholder 1.0.
- Widths (block_q) are 32, 64, 96, 128, 192, 256. For `block_kv >= 1024` only
  128 and 256 remain; other widths are not multiples of the 128-row lane tile
  and were measured 27% to 2.5x slower.
- Width caps: `block_q = 256` only up to `block_kv = 512`, and
  `block_q = 192` up to 640. Measured, 256 at 640/768 is 2.7/2.8x slower than
  512; 192 at 640 is 1.2x slower but at 768/896 it is 2.4/2.5x slower. Their
  slow cells are also excluded from the fit.
- Row groups are 64, 128, 256, 512, 1024. Each `(block_q, block_kv)` is first
  filtered by the VMEM budget, then only the feasible grouping with the
  smallest efficiency factor is kept. 256 is feasible over the pruned lattice
  and has h(256) = 1, so the choice is always `group_rows = 256`; larger
  groupings are used only when 256 does not fit.
- The cheapest configuration wins; ties take the longer `block_kv`, then the
  wider `block_q`.
- Width is also capped by `next_pow2(Qmax)`: a batch of short chunks is not
  priced against a q block it cannot fill.

## Measurements (TPU v6e, bfloat16)

Every cell uses 3 warmup calls and 5 timed samples, with the median reported;
cells over the VMEM budget are skipped. The fit uses 103 cells from 10
(page, q_lens, kv_lens) groups: pages 128/256/512/1024, both full-prompt
(q_len = kv_len) and chunked prefill (q_len < kv_len), lengths 512 to 16384.

e(block_kv), in 1e-9 ms per score element:

| K | 128 | 256 | 384 | 512 | 640 | 768 | 896 | 1024 | 1152 | 1280 | 1408 | 1536 | 1664 | 1792 | 1920 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e | 0 | 0.182 | 1.668 | 1.430 | 2.205 | 1.778 | 2.356 | 2.075 | 2.451 | 2.856 | 2.544 | 6.731 | 5.919 | 6.520 | 5.930 | 5.948 |

Two cliffs drive the pruning:

- e at block_kv 1536/2048 is about 3x e at 1024. At q = kv = 8192, batch 2 and
  block_q 128: K1024 is 6.86 ms, K1536 is 16.00 ms and K2048 is 14.19 ms.
  Larger blocks do not recover either: K2560/3072/3584/4096 are
  13.50/12.70/12.18/11.64 ms, still far slower than K1024. The maximum block
  is therefore 2048, and 1536..2048 stay in the table at their measured cost.
- The width cliff is described above. With block_q = 128, K512/768/1024 all
  run around 7 ms and are the main choice for long contexts.

Row-group measurements on the same workload, block_q and block_kv:

- `group_rows = 256` is fastest in 7 of 8 multi-group cells and ties the
  fastest in the eighth (0.1% apart);
- 128 has a codegen problem: on c_long16k with block_q 128 / block_kv 1024 it
  takes 44.57 ms, while 256 takes 13.33 ms on the same cell. Its fit cells are
  excluded and the table uses the clean K512 ratio, 1.31;
- 64 is 2.5 to 5.6x slower; 512/1024 are a median 3.5%/3.5% slower, worst
  6.1%/7.5%.

On the 103 cells with `group_rows = 256`, the fit's mean relative error is
3.51% and its maximum is 12.66%. Selecting with the fitted model on the same
cells gives a maximum group regret of 1.76% (u8192p128 picks K768, 1.76%
behind the measured best K1024; every other group picks the measured best).
These are in-sample numbers, not the acceptance criterion; the independent
held-out set is below.

## Calibration and fit

The fit splits the formula into three blocks and alternates them for 12
rounds:

1. linear terms and e(K): fix p and h, solve a non-negative least squares
   (NNLS) problem on relative error;
2. h(group_rows): fix the other terms, weighted least squares on measured
   time, with h(256) = 1 pinned;
3. p(block_q): the same weighted least squares, with p(128) = 1 pinned.

Only `group_rows = 256` cells enter the linear-term, e and p fits, because the
choice always takes 256 and that is the decision surface being fitted. The
remaining h values use clean measured ratios. VMEM-skipped cells, cells outside
the width cliffs, K > 2048 cells and the pathological `group_rows = 128` cells
are excluded from the linear fit.

## Validation

The validation workloads are four held-out loads with fixed seeds, different
from the calibration set in page width, batch and length shape:

| workload | page | batch | q_lens | kv_lens | chosen | best | chosen ms | best ms | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| v_uniform2k_p128 | 128 | 8 | 2048 x 8 | 2048 x 8 | bq128/K768/rg16 | bq128/K1024/rg16 | 2.1426 | 2.1269 | 0.74% |
| v_spiky16k_p1024 | 1024 | 4 | 2048,2048,1024,512 | 16384,8192,4096,2048 | bq128/K1024/rg16 | same | 5.0261 | 5.0261 | 0.00% |
| v_longtail32k_p512 | 512 | 2 | 2048,512 | 32768,4096 | bq128/K1024/rg16 | same | 6.5436 | 6.5436 | 0.00% |
| v_pareto8k_p512 | 512 | 4 | 2048,1024,512,512 | 8192,4096,2048,1024 | bq128/K1024/rg16 | same | 1.9852 | 1.9852 | 0.00% |

Here bq/K/rg are block_q/block_kv/row_groups. All 547 cells completed. Mean
regret is 0.19% and the maximum is 0.74%, under 2% for every group. The legacy
`max_seqlen` rule (bq128/K512) scores 1.90% / 4.61% / 4.32% / 5.53% on the
same four workloads.

The oracle is not the heuristic's own candidate set: for each workload it is
every legal cell, all page-legal `block_kv` x measured widths x 5 measured row
groupings, with widths 128/256 when K >= 1024. Pruned configurations are added
back as spot checks: 256@640/768 on page 128, 192@1024 for every workload, and
widths 160/224/240 at 512, for 547 cells total. The heuristic's choice is
located in the oracle and compared with the fastest measured cell; pruned
cells are included in the regret.

## Status and limits

- `CALIBRATION["TPU v6e"]["bfloat16"]` holds the constants above and the
  default path uses them. A chip or dtype without an entry keeps the legacy
  rule.
- On four independent workloads and 547 oracle cells, the worst regret is
  0.74% and the mean is 0.19%, meeting the at-most-2% criterion; the legacy
  rule's worst is 5.53% on the same set.
- Calibration was done on one v6e (`vmem_limit_bytes()` is 100 MiB and the
  candidate budget is 0.75 of it, with a bf16 cache). A different chip,
  compiler or cache dtype needs a new calibration run and a new `CALIBRATION`
  entry.
- K > 2048 is excluded because measurements do not recover. If the kernel's
  long-block path is fixed, the block must be re-measured.
- The prior decides what is optimal on average. If production traffic differs
  substantially from `WORKLOAD_PRIOR`, replace the prior; a workload with a
  known (q_len, kv_len) distribution can use that distribution directly as its
  prior samples.
