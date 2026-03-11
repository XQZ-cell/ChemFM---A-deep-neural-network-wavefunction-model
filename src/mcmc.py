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
from src.utils import utils


def mh_accept(
        state_1: Sequence[jax.Array], 
        state_2: Sequence[jax.Array], 
        lp_ratio: jax.Array, 
        lq_ratio: jax.Array, 
        num_accepts: jax.Array, 
        key: chex.PRNGKey
        ) -> Tuple[List[jax.Array], jax.Array, chex.PRNGKey]:
    """Execute Metropolis-Hastings accept/reject step for state sequences.
    
    Accepts proposed state_2 over current state_1 with probability 
    min(1, exp(lp_ratio + lq_ratio)). Tracks total acceptances across batch.
    
    Args:
        state_1: Current state sequence as a list of arrays. Each array has 
                 shape (batch_size, ...) where first dimension is batch.
        state_2: Proposed state sequence (same structure as state_1).
        lp_ratio: Log probability ratio log p(state_2) - log p(state_1). 
                  Shape: (batch_size,)
        lq_ratio: Log proposal ratio log q(state_1|state_2) - log q(state_2|state_1). 
                  Shape: (batch_size,)
        num_accepts: Cumulative counter of accepted proposals.
        key: Random key for sampling uniform values.
        
    Returns:
        Updated state sequence (either state_2 if accepted, else state_1),
        Updated acceptance counter,
        Updated PRNG key.
    """
    log_ratio = lp_ratio + lq_ratio
    key, subkey = jax.random.split(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=log_ratio.shape))
    del subkey
    cond = rnd < log_ratio
    state_new = []
    for i in range(len(state_1)):
        ndim = state_1[i].ndim
        if ndim == 1:
            # For 1-D array with shape (batch,)
            state_new.append(jnp.where(cond, state_2[i], state_1[i]))
        elif ndim >= 2:
            # For n-D array, add n-1 additional dims to cond for proper broadcasting
            # Example: cond shape (batch,) → (batch, 1, 1, ...) to match state shape
            expand_dims = (...,) + (None,) * (ndim - 1)
            state_new.append(jnp.where(cond[expand_dims], state_2[i], state_1[i]))
        else:
            # Should not happen, but kept for completeness
            raise ValueError(f"Invalid array dimension {ndim} at index {i}")
            
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
        """Metropolis-Hasting algorithm for electronic configuration sampling.
            
        Performs one MCMC step using symmetric Gaussian random walk proposals.
        Accepts proposed electron configurations with probability given by
        the wavefunction squared ratio.
        
        Args:
            params: Network parameters.
            data: Current electron configuration (positions, spins, atoms, charges).
            lp: Current log probability = 2 * log|ψ| for the current configuration.
                Shape (batch_size,).
            num_accepts: Cumulative counter of accepted proposals.
            key: Random key for generating proposals.
            width: Step size scaling factor for Gaussian random walk proposals.
                Controls the variance of the proposal distribution.
                    
        Returns:
            Updated electron configuration with new positions (if accepted),
            Updated log probability for the configuration,
            Updated acceptance counter,
            Updated PRNG key.
        """
        # Old electron positions and log(p)
        x_1, lp_1 = data.positions, lp
        
        # Propose new electron positions using Gaussian random walk
        key, subkey = jax.random.split(key)
        x_2 = x_1 + width * jax.random.normal(subkey, shape=x_1.shape)
        # Compute log probability for new configuration: 2 * log|ψ|
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


def HMC_sample(
        batch_network: Callable,
        batch_momentum_sample: Callable,
        grad_network: Callable,
        params: Dict, 
        data: NetworkInput,
        lp: jax.Array,
        grad_V_prev: jax.Array, 
        num_accepts: jax.Array,
        key: chex.PRNGKey, 
        eps: jax.Array,
        batch_size: int,
        M: jax.Array, 
        M_inver: jax.Array,
        L: int
        ) -> Tuple[NetworkInput, jax.Array, jax.Array, jax.Array, chex.PRNGKey]:
    """Performs one HMC step with leapfrog integration and Metropolis-Hastings acceptance.
    Probability density: p(q,p) ∝ exp(-H(q,p)) = exp(2*log|ψ(q)| - K(p))
    Hamiltonian: H(q,p) = K(p) - 2*log|ψ(q)|
    Acceptance probability: α = min(1, exp(H₁ - H₂))
    
    Args:
        batch_network: Wavefunction network returning log|ψ| for batch inputs.
        grad_network: Function computing gradient of log|ψ| w.r.t. positions.
        batch_K: Kinetic energy function for batch momentum.
        params: Network parameters.
        data: Current electron configuration (positions, spins, atoms, charges).
        lp: Current log probability = 2*log|ψ| (log|ψ|²).
            Shape: (batch_size,)
        grad_V_prev: Cached gradient of V = -2*grad(log|ψ|) at current position.
                     Shape: (batch_size, dim)
        num_accepts: Cumulative counter of accepted proposals.
        key: Random key for momentum sampling.
        eps: Step size for leapfrog integration.
        batch_size: Number of parallel Markov chains.
        M: Mass matrix defining momentum covariance. Shape: (dim, dim)
        M_inver: Inverse of mass matrix for kinetic energy computation.
        L: Number of leapfrog steps per trajectory.
        
    Returns:
        Updated electron configuration with new positions (if accepted),
        Updated log probability (2*log|ψ|) for the configuration,
        Updated gradient of V at new position,
        Updated acceptance counter,
        Updated PRNG key.
    """
    # Extract inputs and parameters
    x_1, lp_1 = data.positions, lp
    dim = x_1.shape[-1]

    if M is None or (isinstance(M, str) and M.lower() == 'none'):
        M = jnp.eye(dim)
        batch_M = jnp.tile(M[None, ...], [batch_size, 1, 1])
    
    # 处理M_inv
    if M_inver is None or (isinstance(M_inver, str) and M_inver.lower() == 'none'):
        M_inver = jnp.eye(dim)
    
    # Make functions
    batch_V = lambda q: - 2.0 * batch_network(params, q, data.spins, data.atoms, data.charges)
    grad_V = lambda q: - 2.0 * grad_network(params, q, data.spins[0], data.atoms[0], data.charges[0])

    def integrate_fun(q, p, grad_V_prev, eps):
        return leapfrog(
            q, p, grad_V_prev, eps, L, grad_V, M_inver)
        
    batch_integrate_fun = jax.vmap(
        integrate_fun,
        in_axes = (0, 0, 0, None), 
        out_axes = (0, 0, 0)
        )
    
    K = lambda p: 0.5 * p.T @ M_inver @ p
    batch_K = jax.vmap(K, in_axes=0, out_axes=0)

    # Momentum resampling
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, batch_size)
    p_1 = batch_momentum_sample(batch_M, subkeys)  # x_1.shape[-2] = batch_size
    del subkey, subkeys

    grad_V_1 = grad_V_prev
    # Integration of Hamiltonian equation, for new proposal
    x_2, p_2, grad_V_2 = batch_integrate_fun(x_1, p_1, grad_V_1, eps)
    lp_2 = - batch_V(x_2)  # # lp = log|psi|^2 = -V

    state_1 = [x_1, lp_1, grad_V_1]
    state_2 = [x_2, lp_2, grad_V_2]

    # Metropolis-Hasting accept/reject step
    lp_ratio = batch_K(p_1) - lp_1 - batch_K(p_2) + lp_2  # = H_1 - H_2, Shape: (batch,)
    lq_ratio = jnp.array(0.0)  # HMC is symmetric proposal
    state_new, num_accepts, key = mh_accept(
        state_1, state_2, lp_ratio, lq_ratio, num_accepts, key)
    x_new, lp_new, grad_V_new = state_new
    data_new = NetworkInput(**(dict(data) | {'positions': x_new}))

    return data_new, lp_new, grad_V_new, num_accepts, key


def make_mcmc_step(
        logabs_network: Callable,
        batch_size: int,
        nsteps: int,
        proposal: str,
        sampler_params: Optional[Dict]
        # max_norm: float = 5.0
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
        # sampler_params should only has key 'max_norm'
        match sampler_params:
            case {'max_norm': max_norm} if len(sampler_params) == 1:
                pass
            case None:
                raise ValueError("sampler_params must be provided for 'langevin' proposal")
            case _:
                raise ValueError(
                    f"sampler_params for 'langevin' must contain exactly one key 'max_norm', "
                    f"got {sampler_params}"
                )
        
        def grad_fn_x(params, x, spins, atoms, charges):
            grad_x = jax.grad(logabs_network, argnums=1)(
                params, x, spins, atoms, charges)
            clipped_grad = utils.clip_norm(grad_x, max_norm)
            
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
    
    elif proposal == 'hamiltonian':
        # Make functions
        M = sampler_params['M']
        M_inver = sampler_params['M_inver']
        L = sampler_params['L']
        max_norm = sampler_params['max_norm']

        def grad_network(params, x, spins, atoms, charges) -> jax.Array:
            grad_x = jax.grad(logabs_network, argnums=1)(
                params, x, spins, atoms, charges)
            clipped_grad = utils.clip_norm(grad_x, max_norm)
            
            return clipped_grad
        
        batch_grad_network = jax.vmap(
            grad_network,
            in_axes = (None, 0, 0, 0, 0), 
            out_axes = 0
            )
        
        batch_momentum_sample = jax.vmap(
            momentum_sample,
            in_axes = (0, 0),
            out_axes = 0
            )

        def inner_fn(params, state, eps):
            return HMC_sample(
                batch_network,
                batch_momentum_sample,
                grad_network,
                params,
                *state,
                eps,
                batch_size,
                M,
                M_inver,
                L
                )
        # initial_state = (data, lp, grad, num_accepts, key)


    else:
        raise ValueError(f"Unknown MCMC move proposal: '{proposal}'.")

    def mcmc_step(
            params: Dict,
            data: NetworkInput,
            key: chex.PRNGKey,
            width: jax.Array
        ) -> Tuple[NetworkInput, jax.Array]:
        """
        Args:
            params:
            data:
            ket:
            width: For HMC, this will be 'eps'.
        
        Returns:

        """
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
        else:  # langevin or hamiltonian
            grad_V = - 2.0 * batch_grad_network(
                params, data.positions, data.spins, data.atoms, data.charges)
            initial_state = (data, lp, grad_V, num_accepts, key)
        
        # MCMC sampling loop
        if proposal == 'hamiltonian':
            final_state = inner_fn(params, initial_state, width)
        else:
            final_state = jax.lax.fori_loop(0, nsteps, sample, initial_state)
        
        if proposal == 'random_walk':
            data_final, lp_final, num_accepts_final, key_final = final_state
        else:  # langevin or hamiltonian
            data_final, lp_final, grad_final, num_accepts_final, key_final = final_state
        
        if proposal == 'hamiltonian':
            pmove = num_accepts_final / batch_size  # For HMC, nsteps = 1
        else:
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
    ) -> Tuple[jax.Array, jax.Array]:
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


def momentum_sample(M: jax.Array, key: chex.PRNGKey) -> jax.Array:
    """Sample a single momentum vector."""
    dim = M.shape[-1]
    mean = jnp.zeros(dim)

    return jax.random.multivariate_normal(key, mean, M)


def leapfrog(
        q: jax.Array,
        p: jax.Array,
        grad_V_prev: jax.Array,
        eps: float,
        L: int,
        grad_V: Callable[jax.Array, jax.Array],
        M_inver: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
    """Performs a single leapfrog integration step for Hamiltonian Monte Carlo.
    
    Implements the leapfrog (or velocity Verlet) integrator to simulate Hamiltonian
    dynamics for HMC sampling. The Hamiltonian is assumed to be H(q,p) = V(q) + K(p),
    where K(p) = 0.5 * p^T M^{-1} p is the kinetic energy. This implementation
    uses a half-step update for momentum at both the beginning and end.
    
    Args:
        q: Current position vector. Shape (dim,).
        p: Current momentum vector. Shape (dim,).
        grad_V_prev: Gradient of the potential energy V at the current position q.
            Shape (dim,).
        eps: Step size for the integration.
        L: Number of integration steps (each of size eps).
        grad_V: Function that computes the gradient of the potential energy V
            at a given position.
        M_inv: Inverse of the mass matrix M. If None, identity matrix is assumed.
            Shape (dim, dim) or None.
            
    Returns:
        A tuple containing:
            - Updated position vector q after L integration steps.
            - Updated momentum vector p after L integration steps.
    """
    def step(i, state):
        q, p, _ = state
        
        q = q + eps * jnp.dot(M_inver, p)
        
        grad_new = grad_V(q)
        
        # 决定使用全步还是半步更新
        is_full_step = i < L - 1  # True 表示全步，False 表示半步
        
        # 全步更新：p - eps * grad_new
        # 半步更新：p - 0.5 * eps * grad_new
        update = jnp.where(is_full_step, 
                           eps * grad_new,  # 全步
                           0.5 * eps * grad_new)  # 半步
        
        p = p - update
        
        return q, p, grad_new
    
    # 初始半步动量更新
    p = p - 0.5 * eps * grad_V_prev
    
    state_final = jax.lax.fori_loop(0, L, step, (q, p, grad_V_prev))
    
    return state_final