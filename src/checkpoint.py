# Standard library imports
import os
import logging
from datetime import datetime
from typing import Tuple, Optional

# Third-party library imports
import jax
import chex
import numpy as np
from orbax import checkpoint as ocp

# Local module imports
from src.optimizer import Pytree, OptimizerState
from src.networks import networks


@chex.dataclass
class CheckpointData:
    iteration: int
    params: Pytree
    data: networks.NetworkInput
    opt_state: OptimizerState
    key: chex.PRNGKey
    mcmc_width: jax.Array
    loss_ewma: float
    pmoves: np.ndarray


def create_save_mngr(
        save_path: Optional[str],
        save_interval_steps: int,
        max_to_keep: int,
        logger: logging.Logger
        ) -> Tuple[ocp.CheckpointManager, str]:
    # Checkpoint manager options
    options = ocp.CheckpointManagerOptions(
        max_to_keep = max_to_keep,
        save_interval_steps = save_interval_steps
        )
    # Maks save path
    current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if save_path:
        save_path = os.path.join(
            save_path, f'checkpoint_{current_time_str}')
    else:
        save_path = os.path.join('.', f'checkpoint_{current_time_str}')
        save_path = os.path.abspath(save_path)
        logger.info(
            f"No save path provided, checkpoints will be saved in {save_path}.")
    # Create checkpoint manager
    ckp_mngr = ocp.CheckpointManager(
            save_path, options=options)
    
    return ckp_mngr, save_path


def restore_from_ckp(
        restore_path: Optional[str],
        logger: logging.Logger,
        ckp_example: Optional[CheckpointData]
        ) -> CheckpointData:
    loaded_ckp = ckp_example  # Checkpoint data template
    if restore_path:
        # Checkpoint checkpoint manager
        options = ocp.CheckpointManagerOptions()  # Default options
        ckp_mngr = ocp.CheckpointManager(
            restore_path, options=options)
        all_steps = ckp_mngr.all_steps()
        if all_steps:  # Found any file in restore_path
            logger.info(f"{len(all_steps)} checkpoint files found in {restore_path}: {all_steps}.")
            # Load checkpoint and restore arguments
            loaded_ckp = ckp_mngr.restore(
                ckp_mngr.latest_step(), 
                args = ocp.args.StandardRestore(ckp_example)
                )  # Type: CheckpointData
            logger.info(f"Load the latest one: {ckp_mngr.latest_step()}")
        else:
            logger.info(f"No checkpoint file found in {restore_path}, start a new training run.")
    else:
        logger.info("No checkpoint restore path provided, start a new training run.")
    
    return loaded_ckp
