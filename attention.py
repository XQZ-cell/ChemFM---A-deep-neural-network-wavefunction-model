import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional
from flax.linen.initializers import zeros, variance_scaling
import numpy as np
import warnings

from folx.experimental.pallas.attention import multi_head_self_attention


class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA) module.

    This module is intended for attending over sequences of vectors.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """
    num_heads: int
    key_size: int
    value_size: int
    dim_out: int
    with_bias: bool = False

    def setup(self):
        # W_q, W_k and W_v
        self.make_Q = nn.Dense(
            self.num_heads * self.key_size,
            use_bias=self.with_bias,
            name="query"
            )
        self.make_K = nn.Dense(
            self.num_heads * self.key_size,
            use_bias=self.with_bias,
            name="key"
            )
        self.make_V = nn.Dense(
            self.num_heads * self.value_size,
            use_bias=self.with_bias,
            name="value"
            )

        # Final projection layer
        self.final_proj = nn.Dense(
            self.dim_out,
            use_bias=self.with_bias,
            name="output"
            )

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray
            ) -> jax.Array:
        """Computes (optionally masked) MHA with queries, keys & values.

        Args:
            query: Embeddings sequence used to compute queries; shape [T', D_q].
            key: Embeddings sequence used to compute keys; shape [T, D_k].
            value: Embeddings sequence used to compute values; shape [T, D_v].
            mask: Optional mask applied to attention weights; shape [H=1, T', T].

        Returns:
            A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [T', D'].
        """
        sequence_len = query.shape[0]

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = self.make_Q(query).reshape((query.shape[0], self.num_heads, self.key_size))  # [T', H, Q=K]
        key_heads = self.make_K(key).reshape((key.shape[0], self.num_heads, self.key_size))  # [T, H, K]
        value_heads = self.make_V(value).reshape((value.shape[0], self.num_heads, self.value_size))  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum("thd,Thd->htT", query_heads, key_heads)  # [H, T', T]
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)

        # Attention weights
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("htT,Thd->thd", attn_weights, value_heads)  # [T', H, V]
        
        attn = jnp.reshape(attn, (sequence_len, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        return self.final_proj(attn)  # [T', D']