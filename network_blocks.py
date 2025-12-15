import jax, chex
from jax import numpy as jnp
from typing import Optional, Tuple


@chex.dataclass
class AuxiliaryNetworkData:
    # Auxiliary data of Network output
    orbitals: jax.Array
    weights: jax.Array


def slogdet(x):
    """Computes sign and log of determinants of matrices.

    This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

    Args:
        x: square matrix.

    Returns:
        sign, (natural) logarithm of the determinant of x.
    """
    if x.shape[-1] == 1:
        if x.dtype == jnp.complex64 or x.dtype == jnp.complex128:
            sign = x[..., 0, 0] / jnp.abs(x[..., 0, 0])
        else:
            sign = jnp.sign(x[..., 0, 0])
            logdet = jnp.log(jnp.abs(x[..., 0, 0]))
    else:
        sign, logdet = jnp.linalg.slogdet(x)

    return sign, logdet


def logdet_matmul(
        xs: jax.Array, 
        w: Optional[jax.Array] = None
        ) -> Tuple[jax.Array, jax.Array, AuxiliaryNetworkData]:
    """Combines determinants and takes dot product with weights in log-domain, 
    and returns determinant weights (normalized by their magnitudes).

    We use the log-sum-exp trick to reduce numerical instabilities.

    Args:
        xs: Full network orbitals in each determinant. Shape (num_dets, nelectrons, nelectrons).
        w: weight of each determinant. If none, a uniform weight is assumed.

    Returns:
        phase_out: Phase of the combined determinant (sign for real, unit complex for complex).
        log_out: Log of the magnitude of the combined determinant (in log domain).
        Auxiliary datas, including network orbitals and determinant weights.
    """
    phase_in, logdet = slogdet(xs)
    
    # log-sum-exp trick
    maxlogdet = jnp.max(logdet)
    det_abs = jnp.exp(logdet - maxlogdet)
    sum_det_abs = jnp.sum(det_abs)
    log_det_sum = jnp.log(sum_det_abs) + maxlogdet
    weights = jnp.exp(logdet - log_det_sum)
    det = phase_in * det_abs
    if w is None:
        result = jnp.sum(det)
    else:
        result = jnp.matmul(det, w)[0]
    
    phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet

    return phase_out, log_out, AuxiliaryNetworkData(
        orbitals = xs, 
        weights = weights
        )
