# Standard library imports
from typing import Callable

# Third-party library imports
import jax
import chex
import numpy as np
from absl import logging

# Local module imports
from src import mcmc
from src.networks import networks


def inference(
        infer_step: Callable,
        iter_final: int,
        params,
        data: networks.NetworkInput,
        key: chex.PRNGKey,
        mcmc_width: jax.Array,
        pmoves: np.ndarray,
        adapt_frequency: int,
        beta: float
    ) -> chex.PRNGKey:
    for i in range(1, iter_final + 1):
        data, key, loss, aux, pmove = infer_step(
            params, data, key, mcmc_width)
        # Update mcmc width
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
        
    return key