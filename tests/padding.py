"""Conversion between padded (batch, seqlen, ...) and packed (total, ...) tokens."""

import jax.numpy as jnp


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool, True means valid.
    Return:
        hidden_states: (total_nnz, ...), the selected tokens.
        indices: (total_nnz,), their positions in the flattened (batch * seqlen).
        cu_seqlens: (batch + 1,) int32, the cumulative sequence lengths.
        max_seqlen_in_batch: int.
        seqused: (batch,) int32, the number of tokens selected in each sequence.
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=jnp.int32)
    indices = jnp.nonzero(attention_mask.reshape(-1))[0]
    max_seqlen_in_batch = int(seqlens_in_batch.max())
    cu_seqlens = jnp.pad(jnp.cumsum(seqlens_in_batch, dtype=jnp.int32), (1, 0))
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return (
        flat[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected
            in attention_mask.
        indices: (total_nnz,), the indices that represent the non-masked tokens of
            the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...), zeros at the masked tokens.
    """
    output = jnp.zeros(
        (batch * seqlen, *hidden_states.shape[1:]), hidden_states.dtype
    )
    output = output.at[indices].set(hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])
