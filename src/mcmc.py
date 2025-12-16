# Standard library imports
from typing import Sequence, Tuple, Dict, Callable, List, Optional

# Third-party library imports
import jax
import chex
import numpy as np
from jax import numpy as jnp

# Local module imports
from src.networks.networks import NetworkInput
from src import constants


def clip_norm(grad, max_norm: float):
    """Clip gradients according to its L2 norm."""
    leaf_sq = jax.tree_util.tree_map(lambda x: jnp.sum(x **2), grad)
    total_sq = jax.tree_util.tree_reduce(lambda x, y: x + y, leaf_sq)
    grad_norm = jnp.sqrt(total_sq)  # L2 norm of grad
    scale = jnp.where(grad_norm > max_norm, max_norm / grad_norm, 1.0)
    clipped_grad = jax.tree_util.tree_map(lambda x: x * scale, grad)
    
    return clipped_grad


def mh_accept(
        state_1: Sequence[jax.Array], 
        state_2: Sequence[jax.Array], 
        lp_ratio: jax.Array, 
        lq_ratio: jax.Array, 
        num_accepts: jax.Array, 
        key: chex.PRNGKey
        ) -> Tuple[List[jax.Array], jax.Array, chex.PRNGKey]:
    """Given states and probabilities, execute MH accept/reject step."""
    log_ratio = lp_ratio + lq_ratio
    key, subkey = jax.random.split(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=log_ratio.shape))
    del subkey
    cond = rnd < log_ratio
    state_new = []
    for i in range(len(state_1)):
        ndim = state_1[i].ndim
        if ndim == 1:
            # For 1D arrays with shape (batch_size,)
            state_new.append(jnp.where(cond, state_2[i], state_1[i]))
        elif ndim == 2:
            # For 2D arrays with shape (batch_size, ...)
            state_new.append(jnp.where(cond[..., None], state_2[i], state_1[i]))
        else:
            # Raise error for unsupported dimensions
            raise ValueError(f"Unsupported array dimension {ndim} for state element at index {i}. "
                             f"Only 1D and 2D arrays are supported.")
    
    num_accepts += jnp.sum(cond)
    
    return state_new, num_accepts, key


def mh_sample(
        batch_network: Callable,
        params: Dict, 
        data: NetworkInput,
        lp: jax.Array,
        num_accepts: jax.Array,
        key: chex.PRNGKey,
        width: jax.Array,
        ) -> Tuple[NetworkInput, jax.Array, jax.Array, chex.PRNGKey]:
    """Metropolis-Hasting algorithm for electronic configuration sampling."""
    # Old electron positions and log(p)
    x_1, lp_1 = data.positions, lp
    
    # New electron positions and log(p)
    key, subkey = jax.random.split(key)
    x_2 = x_1 + width * jax.random.normal(subkey, shape=x_1.shape)
    lp_2 = 2.0 * batch_network(params, x_2, data.spins, data.atoms, data.charges)
    
    # Old and new MCMC states
    state_1 = [x_1, lp_1]
    state_2 = [x_2, lp_2]
    
    # Metropolis-Hasting accept/reject step
    lp_ratio = lp_2 - lp_1  # Shape: (batch,)
    lq_ratio = jnp.array(0.0)  # random walk is symmetric proposal
    state_new, num_accepts, key = mh_accept(
        state_1, state_2, lp_ratio, lq_ratio, num_accepts, key)
    x_new, lp_new = state_new
    data_new = NetworkInput(**(dict(data) | {'positions': x_new}))

    return data_new, lp_new, num_accepts, key


def mh_sample_langevin(
        batch_network: Callable,
        batch_grad_fn: Callable,
        params: Dict, 
        data: NetworkInput,
        lp: jax.Array,
        grad: jax.Array, 
        num_accepts: jax.Array,
        key: chex.PRNGKey,
        width: jax.Array,
        ) -> Tuple[NetworkInput, jax.Array, jax.Array, jax.Array, chex.PRNGKey]:
    """Langevin MCMC algorithm for electronic configuration sampling."""
    # Old electron positions, log(p) and gradient
    x_1, lp_1, grad_1 = data.positions, lp, grad
    
    # New electron positions, log(p) and gradient
    key, subkey = jax.random.split(key)
    mu_1 = x_1 + width**2 * grad_1  # eps^2/2 * ln|psi|^2 = eps^2 * ln|psi|
    x_2 = mu_1 + width * jax.random.normal(subkey, shape=x_1.shape)  # with normal noise
    del subkey
    lp_2 = 2.0 * batch_network(
        params, x_2, data.spins, data.atoms, data.charges)
    grad_2 = batch_grad_fn(
        params, x_2, data.spins, data.atoms, data.charges)
    mu_2 = x_2 + width**2 * grad_2
    
    # q_i_j represents conditional probability q(i|j)
    lq_2_1 = -0.5 * jnp.sum((x_2 - mu_1)**2, axis=-1) / width**2
    lq_1_2 = -0.5 * jnp.sum((x_1 - mu_2)**2, axis=-1) / width**2
    lq_ratio = lq_1_2 - lq_2_1
    
    # Old and new MCMC states
    state_1 = [x_1, lp_1, grad_1]
    state_2 = [x_2, lp_2, grad_2]
    
    # Metropolis-Hasting accept/reject step
    lp_ratio = lp_2 - lp_1
    state_new, num_accepts, key = mh_accept(
        state_1, state_2, lp_ratio, lq_ratio, num_accepts, key)
    x_new, lp_new, grad_new = state_new
    data_new = NetworkInput(**(dict(data) | {'positions': x_new}))

    return data_new, lp_new, grad_new, num_accepts, key


def make_mcmc_step(
        logabs_network: Callable,
        batch_size: int,
        nsteps: int,
        proposal: str,
        max_norm: float = 5.
        ) -> Callable:
    """Construct MCMC step function."""
    batch_network = jax.vmap(
        logabs_network, 
        in_axes = (None, 0, 0, 0, 0), 
        out_axes = 0
        )

    if proposal == 'random_walk':
        def f(params, state, width):
            return mh_sample(
                batch_network, params, *state, width)
        inner_fn = f
    elif proposal == 'langevin':
        def grad_fn_x(params, x, spins, atoms, charges):
            grad_x = jax.grad(logabs_network, argnums=1)(
                params, x, spins, atoms, charges)
            clipped_grad = clip_norm(grad_x, max_norm)
            
            return clipped_grad

        batch_grad_fn = jax.vmap(
            grad_fn_x,
            in_axes = (None, 0, 0, 0, 0), 
            out_axes = 0
            )
        
        def f(params, state, width):
            return mh_sample_langevin(
                batch_network, batch_grad_fn, params, *state, width)
        inner_fn = f
    else:
        raise ValueError(f"Unknown MCMC move proposal: '{proposal}'.")

    def mcmc_step(
        params: Dict,
        data: NetworkInput,
        key: chex.PRNGKey,
        width: jax.Array
        ) -> Tuple[NetworkInput, jax.Array]:
        
        # Body function for lax.fori_loop
        def sample(i, state):
            """Single MCMC step within the loop."""
            return inner_fn(params, state, width)
        
        # Inital state preparation
        lp = 2.0 * batch_network(
            params, data.positions, data.spins, data.atoms, data.charges)
        num_accepts = jnp.array(0)
        
        if proposal == 'random_walk':
            initial_state = (data, lp, num_accepts, key)
        else:  # langevin
            grad = batch_grad_fn(
                params, data.positions, data.spins, data.atoms, data.charges)
            initial_state = (data, lp, grad, num_accepts, key)
        
        # MCMC sampling loop
        final_state = jax.lax.fori_loop(0, nsteps, sample, initial_state)
        
        if proposal == 'random_walk':
            data_final, lp_final, num_accepts_final, key_final = final_state
        else:  # langevin
            data_final, lp_final, grad_final, num_accepts_final, key_final = final_state
        
        pmove = num_accepts_final / (nsteps * batch_size)
        # Device average
        pmove = constants.pmean(pmove)

        return data_final, pmove

    return mcmc_step


def make_pretrain_mcmc_step(
        batch_network: Callable,
        batch_size: int,
        nsteps: int
        ) -> Callable:
    """Construct MCMC step function."""
    def f(params, state, width):
        return mh_sample(
            batch_network, params, *state, width)
    inner_fn = f

    def mcmc_step(
        params: Dict,
        data: NetworkInput,
        key: chex.PRNGKey,
        width: jax.Array
        ) -> Tuple[NetworkInput, jax.Array]:
        
        # Body function for lax.fori_loop
        def sample(i, state):
            """Single MCMC step within the loop."""
            return inner_fn(params, state, width)
        
        # Inital state preparation
        lp = 2.0 * batch_network(
            params, data.positions, data.spins, data.atoms, data.charges)
        num_accepts = jnp.array(0)
        
        initial_state = (data, lp, num_accepts, key)
        # MCMC sampling loop
        final_state = jax.lax.fori_loop(0, nsteps, sample, initial_state)
        data_final, lp_final, num_accepts_final, key_final = final_state
        
        pmove = num_accepts_final / (nsteps * batch_size)
        # Device average
        pmove = constants.pmean(pmove)

        return data_final, pmove

    return mcmc_step



def update_mcmc_width(
        t: int,
        width: jax.Array,
        adapt_frequency: int,
        pmove: jax.Array,
        pmoves: np.ndarray,
        pmove_max: float = 0.55,
        pmove_min: float = 0.5,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Updates the width in MCMC steps.

    Args:
        t: Current step.
        width: Current MCMC width.
        adapt_frequency: The number of iterations after which the update is applied.
        pmove: Acceptance ratio in the last step.
        pmoves: Acceptance ratio over the last N steps, where N is the number of
                steps between MCMC width updates.
        pmove_max: The upper threshold for the range of allowed pmove values
        pmove_min: The lower threshold for the range of allowed pmove values

    Returns:
        width: Updated MCMC width.
        pmoves: Updated `pmoves`.
    """
    t_since_mcmc_update = t % adapt_frequency
    # update `pmoves`; `pmove` should be the same across devices
    pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
    if t > 0 and t_since_mcmc_update == 0:
        if jnp.mean(pmoves) > pmove_max:
            width *= 1.1
        elif jnp.mean(pmoves) < pmove_min:
            width /= 1.1
    return width, pmoves


def leapfrog(
        q: jax.Array, 
        p: jax.Array, 
        grad_V: Callable, 
        M_inv: Optional[jax.Array], 
        eps: float, 
        L: int
        ):
    # LeapFrog method for solving the canonical equation in HMC
    def step(q, p):
        # half step p
        p = p - 0.5 * eps * grad_V(q)
        # full step q
        if M_inv:
            q = q + eps * jnp.dot(M_inv, p)
        else:
            q = q + eps * p  # Assume identity matrix
        # full step p
        p = p - 0.5 * eps * grad_V(q)
        
        return q, p
    
    def body_fun(i, state):
        return step(*state)
    
    init_state = (q, p)
    final_state = jax.lax.fori_loop(0, L, body_fun, init_state)

    return final_state
