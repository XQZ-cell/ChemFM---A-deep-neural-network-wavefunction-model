# Standard library imports
import functools
from typing import Tuple, Union, Any, Dict, Callable, Optional

# Third-party library imports
import jax
import chex
import kfac_jax
import optax
from jax import numpy as jnp
from typing_extensions import Protocol

# Local module imports
from src.networks import networks
from src.loss import loss
from src import constants


Pytree = Any

# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]

OptResults = Tuple[
    Pytree,
    networks.NetworkInput,
    OptimizerState,
    chex.PRNGKey,
    jax.Array,
    loss.AuxiliaryLossData,
    jax.Array
    ]

InferResults = Tuple[
    networks.NetworkInput,
    chex.PRNGKey,
    jax.Array,
    loss.AuxiliaryLossData,
    jax.Array
    ]


class OptStep(Protocol):
    """Protocol for training step functions.
    
    Defines the interface for functions that perform one complete training step,
    combining MCMC sampling and parameter optimization.
    """
    
    def __call__(
        self,
        params: Pytree,
        data: networks.NetworkInput,
        opt_state: OptimizerState,
        key: chex.PRNGKey,
        mcmc_width: jax.Array
    ) -> OptResults:
        """Execute one MCMC + optimization training step.
        
        Performs MCMC sampling followed by a parameter update.
        
        Args:
            params: Network parameters (replicated to all devices).
            data: MCMC configurations, spins, and atomic positions (replicated).
            opt_state: Optimizer internal state (replicated).
            key: JAX RNG state with device axis (replicated).
            mcmc_width: MCMC proposal width (replicated).

        Returns:
            Tuple of 7 elements:
            - updated_params: Network parameters after gradient update
            - updated_data: MCMC configurations drawn using input parameters
            - new_opt_state: Updated optimizer state
            - new_key: New random key with device axis
            - loss: Energy averaged over MCMC configurations
            - aux_data: Additional loss statistics
            - pmove: MCMC acceptance probability
        """

class InferStep(Protocol):
    """Protocol for inference step functions.
    
    Defines the interface for functions that perform one complete training step,
    combining MCMC sampling and parameter optimization.
    """
    
    def __call__(
        self,
        params: Pytree,
        data: networks.NetworkInput,
        key: chex.PRNGKey,
        mcmc_width: jax.Array
    ) -> InferResults:
        """Execute one MCMC + optimization training step.
        
        Performs MCMC sampling followed by a parameter update.
        
        Args:
            params: Network parameters (replicated to all devices).
            data: MCMC configurations, spins, and atomic positions (replicated).
            key: JAX RNG state with device axis (replicated).
            mcmc_width: MCMC proposal width (replicated).

        Returns:
            Tuple of 5 elements:
            - updated_data: MCMC configurations drawn using input parameters
            - new_key: New random key with device axis
            - loss: Energy averaged over MCMC configurations
            - aux_data: Additional loss statistics
            - pmove: MCMC acceptance probability
        """


def make_infer_step(
        mcmc_step: Callable, 
        loss_fn: Callable
        ) -> InferStep:
    @functools.partial(constants.pmap, donate_argnums=1)
    def infer_step(
            params: Pytree,
            data: networks.NetworkInput,
            key: chex.PRNGKey,
            mcmc_width: jax.Array
            ) -> InferResults:
        # MCMC sampling
        key, mcmc_key = jax.random.split(key)
        data, pmove = mcmc_step(
            params, data, mcmc_key, mcmc_width)
        del mcmc_key
        # Parameter optimization
        loss, aux = loss_fn(params, data)
    
        return data, key, loss, aux, pmove

    return infer_step


def make_optax_train_step( 
        mcmc_step: Callable, 
        loss_fn: Callable,
        optimizer: optax.GradientTransformation
        ) -> OptStep:
    @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
    def train_step(
            params: Pytree, 
            data: networks.NetworkInput,
            opt_state: optax.OptState,
            key: chex.PRNGKey,
            mcmc_width: jax.Array
            ) -> OptResults:
        # MCMC sampling
        key, mcmc_key = jax.random.split(key)
        data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)
        del mcmc_key
        # Parameter optimization
        loss_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, aux), grad = loss_and_grad(params, data)
        grad = constants.pmean(grad)  # Device average gradient
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
    
        return params, data, opt_state, key, loss, aux, pmove

    return train_step


def make_kfac_train_step(
        mcmc_step: Callable, 
        damping: float, 
        optimizer: kfac_jax.Optimizer
        ) -> OptStep:
    """Create K-FAC training step function.
    
    Note: Do NOT stage (jit/pmap) optimizer/loss - kfac_jax already does this.
    
    Args:
        mcmc_step: MCMC sampling function.
        damping: K-FAC damping parameter.
        optimizer: K-FAC optimizer object.

    Returns:
        Function that executes one K-FAC training step.
    """
    # Parallelize MCMC and replicate shared parameters
    mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(damping))

    def train_step(
            params: Pytree, 
            data: networks.NetworkInput,
            opt_state: kfac_jax.Optimizer.State, 
            key: chex.PRNGKey,
            mcmc_width: jax.Array
            ) -> OptResults:
        """Execute one K-FAC training step.

        Note: All inputs must be replicated to all devices.
        
        Args:
            params: Network parameters (replicated).
            data: Network inputs (replicated).
            opt_state: K-FAC optimizer state (replicated).
            key: Random key with device axis (replicated).
            mcmc_width: MCMC step size (replicated).

        Returns:
            Tuple of 7 elements:
            - updated_params: Network parameters after gradient update
            - updated_data: MCMC configurations drawn using input parameters
            - new_opt_state: Updated optimizer state
            - new_sharded_key: New random key with device axis
            - loss: Energy averaged over MCMC configurations
            - aux_data: Additional loss statistics
            - pmove: MCMC acceptance probability
        """
        # MCMC sampling
        sharded_key, mcmc_keys = kfac_jax.utils.p_split(key)
        data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)
        del mcmc_keys
        
        # K-FAC optimization
        sharded_key, loss_keys = kfac_jax.utils.p_split(sharded_key)
        params, opt_state, stats = optimizer.step(
            params=params, 
            state=opt_state,
            rng=loss_keys,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping
            )
        del loss_keys

        return params, data, opt_state, sharded_key, stats['loss'], stats['aux'], pmove
    
    return train_step


def init_optim(
        cfg_optim,
        lr_schedule: Callable[int, jax.Array],
        mcmc_step: Callable,
        loss_fn: Optional[Callable],
        params: Pytree,
        data: Optional[networks.NetworkInput],
        sharded_key: Optional[chex.PRNGKey]
        ) -> Tuple[OptStep, OptimizerState]:
    ############################################################ 
    #                                                          #
    #                      Adam Optimizer                      #   
    #                                                          #
    ############################################################
    if cfg_optim.optimizer == 'adam':
        optimizer = optax.chain(
        optax.scale_by_adam(**cfg_optim.adam),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.)
        )
        opt_state = constants.pmap(optimizer.init)(params)
        # Make optax training step
        train_step = make_optax_train_step(
            mcmc_step, 
            loss_fn, 
            optimizer
            )
    ############################################################ 
    #                                                          #
    #                      K-FAC Optimizer                     #
    #                                                          #
    ############################################################
    elif cfg_optim.optimizer == 'kfac':
        val_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        optimizer = kfac_jax.Optimizer(
            val_and_grad,
            l2_reg = cfg_optim.kfac.l2_reg,
            norm_constraint = cfg_optim.kfac.norm_constraint,
            value_func_has_aux = True,
            value_func_has_rng = False,
            learning_rate_schedule = lr_schedule,
            curvature_ema = cfg_optim.kfac.cov_ema_decay,
            inverse_update_period = cfg_optim.kfac.invert_every,
            min_damping = cfg_optim.kfac.min_damping,
            num_burnin_steps = 0,
            register_only_generic = cfg_optim.kfac.register_only_generic,
            estimation_mode = 'fisher_exact',
            multi_device = True,
            pmap_axis_name = constants.PMAP_AXIS_NAME
            )
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        opt_state = optimizer.init(params, subkeys, data)
        del subkeys
        # Make K-FAC training step
        damping = cfg_optim.kfac.damping
        train_step = make_kfac_train_step(
            mcmc_step, 
            damping, 
            optimizer
            )
    
    else:
        raise NotImplementedError(f"Unknown optimizer: {cfg_optim.optimizer}.")

    return train_step, opt_state
    
