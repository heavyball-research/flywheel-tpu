"""flash_attn_bwd against jax.vjp of a naive reference.

The kernel is fed (o, lse) from that same reference forward, the way a real
forward pass hands its residuals to the backward.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu.pallas import flash_bwd
from flywheel_tpu.pallas.flash_bwd import flash_attn_bwd
from tests.reference import head_major_ref, rel_err

INTERPRET = jax.default_backend() != "tpu"


def check_backward(dtype, causal, num_heads=3, seqlen=256, head_dim=64,
                   block=128, block_kv=None, block_q_compute=None,
                   block_kv_compute=None, head_fold=1, head_dim_v=None,
                   kv_len=None, nheads_k=None):
  """Runs one backward configuration against the reference.

  seqlen and kv_len are the real lengths: each pads to a multiple of 128 and
  the causal offset comes from the real pair, as in flash_attn_func.
  """
  head_dim_v = head_dim if head_dim_v is None else head_dim_v
  kv_len = seqlen if kv_len is None else kv_len
  nheads_k = num_heads if nheads_k is None else nheads_k
  scale = 1.0 / math.sqrt(head_dim)
  causal_offset = kv_len - seqlen
  q_padded_len = (seqlen + 127) // 128 * 128
  kv_padded_len = (kv_len + 127) // 128 * 128

  def padded_normal(key, heads, real_len, padded_len, dim):
    real_rows = jax.random.normal(key, (heads, real_len, dim), dtype)
    return jnp.pad(real_rows, ((0, 0), (0, padded_len - real_len), (0, 0)))

  key_q, key_k, key_v, key_do = jax.random.split(jax.random.PRNGKey(0), 4)
  q = padded_normal(key_q, num_heads, seqlen, q_padded_len, head_dim)
  k = padded_normal(key_k, nheads_k, kv_len, kv_padded_len, head_dim)
  v = padded_normal(key_v, nheads_k, kv_len, kv_padded_len, head_dim_v)
  do = padded_normal(key_do, num_heads, seqlen, q_padded_len, head_dim_v)

  # Note (david): expanding k/v inside the reference makes jax.vjp against the
  # unexpanded k/v return the group-summed GQA dk/dv.
  def reference(q, k, v):
    group = num_heads // nheads_k
    return head_major_ref(
        q, jnp.repeat(k, group, axis=0), jnp.repeat(v, group, axis=0), causal,
        scale, causal_offset)

  o_ref, reference_vjp, lse = jax.vjp(reference, q, k, v, has_aux=True)
  grads_ref = reference_vjp(do.astype(jnp.float32))
  grads = flash_attn_bwd(
      q, k, v, o_ref.astype(dtype), lse, do,
      causal=causal, causal_offset=causal_offset, softmax_scale=scale,
      block=block, block_kv=block_kv, block_q_compute=block_q_compute,
      block_kv_compute=block_kv_compute, head_fold=head_fold,
      interpret=INTERPRET,
  )
  # Note (david): on TPU the MXU runs f32 matmuls as bf16 passes by default,
  # so even f32 gradients only get to ~2e-3.
  tol = 5e-3 if dtype == jnp.float32 else 3e-2
  for name, grad, grad_ref in zip("qkv", grads, grads_ref):
    assert grad.shape == grad_ref.shape, (name, grad.shape, grad_ref.shape)
    assert rel_err(grad, grad_ref) < tol, (name, rel_err(grad, grad_ref))
  # Note (david): rows with no visible key must get exact zeros, including
  # whole q blocks the schedule never visits.
  num_empty_rows = max(0, -causal_offset) if causal else 0
  assert jnp.all(grads[0][:, :num_empty_rows] == 0)


@pytest.mark.parametrize(
    "dtype,config",
    [
        pytest.param(jnp.float32, {}, id="f32"),
        pytest.param(jnp.bfloat16, {}, id="bf16"),
        pytest.param(jnp.bfloat16, dict(seqlen=512), id="multi_block"),
        pytest.param(jnp.bfloat16, dict(seqlen=512, block=512,
                                        block_kv_compute=128),
                     id="kv_subtiles"),
        pytest.param(jnp.bfloat16, dict(seqlen=512, block=512,
                                        block_q_compute=128,
                                        block_kv_compute=128),
                     id="q_and_kv_compute_tiles"),
        # Note (david): 8 x 8 blocks put most (kv, q) pairs strictly below the
        # diagonal, where the kernel skips the causal mask.
        pytest.param(jnp.float32, dict(num_heads=2, seqlen=1024, block_kv=128),
                     id="below_diagonal"),
        pytest.param(jnp.float32, dict(num_heads=4, block=256, head_fold=None,
                                       head_dim_v=128),
                     id="head_dim_v"),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_bwd_dense(causal, dtype, config):
  check_backward(dtype, causal, **config)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_bwd_single_block_head_fold(causal, dtype):
  # Note (david): 6 heads folded 3-wide is two groups with a non-power-of-2
  # fold.
  check_backward(dtype, causal, num_heads=6, seqlen=512, block=512,
                 head_fold=3)


@pytest.mark.parametrize(
    "q_len,kv_len,causal",
    [
        (128, 256, False),
        (128, 256, True),
        (256, 128, False),
        # Note (david): the fully masked rows carry lse == -inf, which NaNs
        # dk/dv without the kernel's lse guard.
        (256, 128, True),
        (200, 120, True),
        (120, 333, True),
        # Note (david): the whole leading q block sits below the diagonal and
        # never enters the schedule, so only the explicit dq zero-fill covers it.
        (2048, 128, True),
    ],
)
def test_bwd_rect(q_len, kv_len, causal):
  check_backward(jnp.bfloat16, causal, seqlen=q_len, kv_len=kv_len,
                 block=None, head_fold=None)


@pytest.mark.parametrize(
    "dtype,causal,config",
    [
        *[(jnp.float32, causal, dict(nheads_k=nheads_k, seqlen=512))
          for causal in (False, True) for nheads_k in (2, 1)],
        *[(jnp.bfloat16, causal, dict(nheads_k=2, seqlen=512))
          for causal in (False, True)],
        (jnp.float32, True, dict(nheads_k=2, kv_len=512)),
        (jnp.float32, False, dict(nheads_k=2, head_dim_v=128)),
    ],
)
def test_bwd_gqa(dtype, causal, config):
  check_backward(dtype, causal, num_heads=4, head_fold=None, **config)


@pytest.mark.parametrize(
    "dtype,causal,nheads_k,kv_len",
    [
        (jnp.float32, False, 4, None),
        (jnp.float32, True, 4, None),
        (jnp.float32, False, 2, None),
        (jnp.float32, True, 2, None),
        (jnp.bfloat16, True, 4, None),
        # Note (david): q 640 against kv 256 leaves chunk 0 wholly below the
        # diagonal and never built, and chunk 1 straddling it.
        (jnp.float32, True, 4, 256),
    ],
)
def test_bwd_q_chunked(monkeypatch, dtype, causal, nheads_k, kv_len):
  # Note (david): the dq_acc gate then fits exactly two q blocks of dq, so the
  # 5-block build must run as q-row chunks, or it raises.
  q_heads_per_kv_head = 4 // nheads_k
  monkeypatch.setattr(flash_bwd, "DQ_ACC_VMEM_LIMIT_BYTES",
                      2 * 128 * q_heads_per_kv_head * 128 * 4)
  check_backward(
      dtype, causal, num_heads=4, nheads_k=nheads_k, seqlen=640,
      kv_len=kv_len, block=128, block_kv=128,
      head_fold=1 if q_heads_per_kv_head == 1 else None)
