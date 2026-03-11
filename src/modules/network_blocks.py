# Standard library imports
from typing import Optional, Tuple, Sequence, Dict

# Third-party library imports
import jax
import chex
from jax import numpy as jnp
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryNetworkData:
    # Auxiliary data of Network output
    orbitals: jax.Array
    weights: jax.Array


class Filter(Protocol):
    def __call__(
        self,
        param: float,
        r: float
        ) -> float:
        """Spatial filter function.
        
        Args:
            params: filter parameter.
            r: spatial distance in real space.
        
        Return:
            Rescaled spatial distance.
        """

exp_filter = lambda beta, r: beta * jnp.exp(- beta * r)
sigmoid_filter = lambda gamma, r: 1 / (gamma * (1 + jnp.exp(r - gamma)))

def apply_filter(
        r: jax.Array, 
        xyz: Optional[jax.Array], 
        filter: Filter, 
        param: float
        ) -> jax.Array:
    """Using the filter to rescale the real space distances and vectors.
        
    Args:
        r: distance between particle i and j. Shape: (M, N, 1).
        xyz: position vector difference between particle i and j. Shape: (M, N, ndim).
        filter: filter function.
        param: filter parameter.
        
    Return:
        Distance features rescaled by given filters. If xyz, rescaled r and xyz will be 
        concatenated, resulting in shape (M, N, 1+ndim). If no xyz, only rescaled r 
        with shape (M, N, 1).
    """
    rescaled_r_aa = filter(param, r)  # Shape: (M, N, 1)
    if xyz is not None:
        rescaled_aa = rescaled_r_aa * xyz  # Product of (M, N, 1) and (M, N, ndim)
        final_feat = jnp.concatenate((rescaled_r_aa, rescaled_aa), -1)
    else:
        final_feat = rescaled_r_aa

    return final_feat  # Shape: (M, N, 1+ndim)


def apply_multi_filters(
        r: jax.Array, 
        xyz: Optional[jax.Array],
        filter_dict: Dict[Filter, float]
        ) -> jax.Array:
    """Apply multiple filters to rescale the real space distances and vectors.
    
    Args:
        r: distance between particle i and j. Shape: (M, N, 1).
        xyz: position vector difference between particle i and j. Shape: (M, N, ndim).
        filter: A list of nfilter filter functions.
        param: A list of nfilter filter parameters.
        
    Return:
        Distance features rescaled by given filters. If xyz, rescaled r and xyz will be 
        concatenated, resulting in shape (M, N, (1+ndim)*nfilters). If no xyz, only 
        rescaled r with shape (M, N, nfilters).
    """
    # Features rescaled by different filters
    features = [
        apply_filter(r, xyz, filter_func, param)
        for filter_func, param in filter_dict.items()
        ]
    
    return jnp.concatenate(features, axis=-1) if features else jnp.array([])


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
