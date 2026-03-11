# Standard library imports
from typing import Tuple

# Third-party library imports
import jax
from jax import numpy as jnp


def make_real_space_vector_diff(
        pos_1: jax.Array,
        pos_2: jax.Array,
        ndim: int
        ) -> Tuple[jax.Array, jax.Array]:
    """Calculate differences between two sets of position vectors in real space.
    
    Args:
        pos_1: position vectors in real space. Shape (N, ndim).
        pos_2: position vectors in real space. Shape (M, ndim).
        ndim: dimension of system. Change only with caution.

    Returns:
        Shape (N, M, ndim).
    """
    assert pos_1.shape[-1] == ndim and pos_2.shape[-1] == ndim

    vec_diff = pos_2[None, :, :] - pos_1[:, None, :]  # Shape: (N, M, ndim)
    r = jnp.linalg.norm(vec_diff, axis=-1, keepdims=True)  # Shpe: (N, M, 1)

    return vec_diff, r


def clip_norm(
        grad: jax.Array, 
        max_norm: float
        ) -> jax.Array:
    """Clip gradient PyTree to have L2 norm ≤ max_norm.
    
    Scales all elements by (max_norm / current_norm) if norm exceeds max_norm.
    
    Args:
        grad: Gradient PyTree to be clipped.
        max_norm: Maximum allowed L2 norm.
        
    Returns:
        Clipped gradient PyTree with same structure as input.
    """
    leaf_sq = jax.tree_util.tree_map(lambda x: jnp.sum(x **2), grad)
    total_sq = jax.tree_util.tree_reduce(lambda x, y: x + y, leaf_sq)
    grad_norm = jnp.sqrt(total_sq)  # L2 norm of grad
    scale = jnp.where(grad_norm > max_norm, max_norm / grad_norm, 1.0)
    clipped_grad = jax.tree_util.tree_map(lambda x: x * scale, grad)
    
    return clipped_grad