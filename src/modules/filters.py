# Third-party library imports
import jax
from jax import numpy as jnp

'''
class filter(Protocol):

  def __call__(self, r: jax.Array, beta: float) -> jax.Array:
    """Filter

    Args:
        r:
        beta:
      
    """
'''

def exp_filter(r: jax.Array, beta: float) -> jax.Array:
    """Exponential filter: r -> βe^(-βr)"""
    return beta * jnp.exp(- beta * r)


def sigmoid_filter(r: jax.Array, gamma: float) -> jax.Array:
    """Exponential filter: r -> 1 / γ(1+e^(r-γ))"""
    return 1.0 / (gamma * (1.0 + jnp.exp(r - gamma)))