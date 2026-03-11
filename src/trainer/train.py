# Standard library imports
from typing import Any

# Third-party library imports
import jax
import chex
import numpy as np
from absl import logging
from orbax import checkpoint as ocp

# Local module imports
from src import mcmc, optimizer, checkpoint
from src.networks import networks


def train(
        train_step: optimizer.OptStep,
        iter_start: int,
        iter_final: int,
        params: Any,
        data: networks.NetworkInput,
        opt_state: optimizer.OptimizerState,
        key: chex.PRNGKey,
        mcmc_width: jax.Array,
        pmoves: np.ndarray,
        adapt_frequency: int,
        loss_ewma: float,
        beta: float,
        save_interval_steps: int,
        save_mngr: ocp.CheckpointManager,
        show_params: bool
    ) -> chex.PRNGKey:
    for i in range(iter_start + 1, iter_final + 1):
        # MCMC sampling and parameter optimization
        params, data, opt_state, key, loss, aux, pmove = train_step(
            params, data, opt_state, key, mcmc_width)  # Type: optimizer.StepResults
        
        # Update MCMC width
        mcmc_width, pmoves = mcmc.update_mcmc_width(
            i, mcmc_width, adapt_frequency, pmove, pmoves)
            
        loss = jax.device_get(loss)[0].item()  # Type: float
        pmove = jax.device_get(pmove)[0].item()
        # EWMA
        if i == 1:
            loss_ewma = loss
        else:
            loss_ewma = beta * loss_ewma + (1 - beta) * loss
        # Information logging
        logging.info(
            f"Step {i}: E = {loss:.6f}, E_ewma = {loss_ewma:.6f}, pmove = {pmove:.4f}")
        # Output specific parameters
        if show_params and i % 500 == 0: 
            params_local = jax.device_get(params)
            ee_anti = params['params']['PsiformerOrbitals']['jastrow_fn']['ee_anti'][0].item()
            ee_par = params['params']['PsiformerOrbitals']['jastrow_fn']['ee_par'][0].item()
            logging.info(f"Jastrow parameters: ee_anti = {ee_anti}, ee_par = {ee_par}")
        # Checkpoint saving
        if i == 1 or i % save_interval_steps == 0:
            # Replicate data on all devices to local device
            params_local = jax.device_get(params)
            data_local = jax.device_get(data)
            opt_state_local = jax.device_get(opt_state)
            sharded_key_local = jax.device_get(key)
            mcmc_width_local = jax.device_get(mcmc_width)
            pmoves_local = jax.device_get(pmoves)
            ckp_to_save = checkpoint.CheckpointData(
                iteration = i,
                params = params_local,
                data = data_local,
                opt_state = opt_state_local,
                key = sharded_key_local,
                mcmc_width = mcmc_width_local,
                loss_ewma = loss_ewma,
                pmoves = pmoves_local
                )
            save_mngr.save(step=i, args=ocp.args.StandardSave(ckp_to_save))
        '''
        if aux_data.weights != None:
            logger.info(f"Average det weights: {aux_data.weights}")
        '''
    save_mngr.wait_until_finished()
    
    return key