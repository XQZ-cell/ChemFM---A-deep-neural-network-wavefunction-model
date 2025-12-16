# Standard library imports
import os
import sys
import time
import functools
from datetime import datetime
from typing import Optional, Mapping, Sequence, Tuple, Dict, Union, Callable

# Third-party library imports
import jax
import chex
import kfac_jax
import optax
import numpy as np
import ml_collections
from jax import numpy as jnp
from flax import linen as nn
from orbax import checkpoint as ocp

# Local module imports
from src.networks import networks
from src.modules import envelopes
from src.modules import jastrows
from src import mcmc
from src.loss import local_energy
from src.loss import loss as qmc_loss
from src import log
from src import pretrain
from src import constants
from src import optimizer
from src import checkpoint
from src.utils import system


def setup_devices_and_batch(
        total_batch_size: int,
        logger
        ) -> Tuple[int, int]:
    """Get device information and calculate batch size."""
    # Show device information
    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"Found {num_devices} available JAX devices:")
    for i, device in enumerate(devices):
        device_info = {
            "index": i,
            "device_id": device.id,
            "device_type": device.device_kind,
            "platform": device.platform,
            "memory_size": "Unknown"
            }
        info_str = (
            f"Device {device_info['index']}: "
            f"[Type: {device_info['device_type']}, "
            f"ID: {device_info['device_id']}, "
            f"Platform: {device_info['platform']}, "
            f"Memory: {device_info['memory_size']}]"
            )
        logger.info(info_str)
    # Get leading data dimensions
    assert total_batch_size % num_devices == 0, (
        f"Batch size must be divisible by number of devices. "
        f"Got {total_batch_size} and {num_devices}."
        )
    device_batch_size = total_batch_size // num_devices

    return (num_devices, device_batch_size)


# Functions for initialization of electron positions
def _assign_spin_configuration(
        nalpha: int, 
        nbeta: int, 
        batch_size: int = 1
        ) -> jax.Array:
    """Returns the spin configuration for a fixed spin polarisation."""
    spins = jnp.concatenate((jnp.ones(nalpha), - jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))

def init_electrons(  # pylint: disable=dangerous-default-value
        key: chex.PRNGKey,
        molecule: Sequence[system.Atom],
        electrons: Sequence[int],
        batch_size: int,
        init_width: float,
        core_electrons: Mapping[str, int] = {},
        max_iter: int = 10_000,
        ) -> Tuple[jax.Array, jax.Array]:
    """Initializes electron positions around each atom.

    Args:
        key: JAX RNG state.
        molecule: system.Atom objects making up the molecule.
        electrons: tuple of number of alpha and beta electrons.
        batch_size: total number of MCMC configurations to generate across all
                    devices.
        init_width: width of (atom-centred) Gaussian used to generate initial
                    electron configurations.
        core_electrons: mapping of element symbol to number of core electrons
                        included in the pseudopotential.
        max_iter: maximum number of iterations to try to find a valid initial
                electron configuration for each atom. If reached, all electrons are
                initialised from a Gaussian distribution centred on the origin.

    Returns:
        array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
        positions in the initial MCMC configurations and ndim is the dimensionality
        of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
        of spin configurations, where 1 and -1 indicate alpha and beta electrons
        respectively.
    """
    niter = 0
    total_electrons = sum(atom.charge - core_electrons.get(atom.symbol, 0)
            for atom in molecule)
    if total_electrons != sum(electrons):
        if len(molecule) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
    else:
        atomic_spin_configs = [
                (atom.element.nalpha - core_electrons.get(atom.symbol, 0) // 2,
                    atom.element.nbeta - core_electrons.get(atom.symbol, 0) // 2)
                for atom in molecule
                ]
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while (
                tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons
                and niter < max_iter
                ):
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            atomic_spin_configs[i] = nbeta, nalpha
            niter += 1

    if tuple(sum(x) for x in zip(*atomic_spin_configs)) == electrons:
        # Assign each electron to an atom initially.
        electron_positions = []
        for i in range(2):
            for j in range(len(molecule)):
                atom_position = jnp.asarray(molecule[j].coords)
                electron_positions.append(
                        jnp.tile(atom_position, atomic_spin_configs[j][i]))
        electron_positions = jnp.concatenate(electron_positions)
    else:
        logger.warning(
                'Failed to find a valid initial electron configuration after %i'
                ' iterations. Initializing all electrons from a Gaussian distribution'
                ' centred on the origin. This might require increasing the number of'
                ' iterations used for pretraining and MCMC burn-in. Consider'
                ' implementing a custom initialisation.',
                niter,
                )
        electron_positions = jnp.zeros(shape=(3*sum(electrons),))
    # Create a batch of configurations with a Gaussian distribution about each
    # atom.
    key, subkey = jax.random.split(key)
    electron_positions += (
            jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
            * init_width
            )

    electron_spins = _assign_spin_configuration(
            electrons[0], electrons[1], batch_size
            )

    return electron_positions, electron_spins


def train(cfg: ml_collections.ConfigDict):
    # Create logger
    logger = log.setup_logging()
    logger.info(cfg)  # Show cfg information

    # Random seed and key
    if cfg.debug.deterministic:
        seed = 23
        logger.info(f"DEBUG mode enabled, using a fixed random number seed={seed}. "
                    f"This will be overridden when checkpoint loaded.")
    else:
        seed = int(time.time() * 1e6)
    key = jax.random.PRNGKey(seed)

    ############################################################ 
    #                                                          #
    #                    Device Information                    #
    #                                                          #
    ############################################################
    total_batch_size = cfg.batch_size
    leading_data_shape = setup_devices_and_batch(total_batch_size, logger)
    device_batch_size = leading_data_shape[1]

    # Get system information
    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])  # Shape: (natoms, ndim)
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])  # Shape: (natoms,)
    nspins = cfg.system.electrons  # A Tuple of (nalpha, nbeta)

    # Generate batched atomic configurations
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)  # Shape: (num_devices, device_batch_size, natoms, ndim)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)  # Shape: (num_devices, device_batch_size, ne=nalpha+nbeta)

    # Initialization of electron configurations
    key, subkey = jax.random.split(key)
    batch_pos, batch_spins = init_electrons(
        subkey,
        molecule = cfg.system.molecule,
        electrons = nspins,
        batch_size = total_batch_size,
        init_width = cfg.mcmc.init_width,
        core_electrons = {},
        max_iter = 10_000
        )
    del subkey

    batch_pos = batch_pos.reshape(leading_data_shape + (-1,))  # Shape: (num_devices, device_batch_size, ne*ndim)
    batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
    batch_spins = batch_spins.reshape(leading_data_shape + (-1,))  # Shape: (num_devices, device_batch_size, ne)
    batch_spins = kfac_jax.utils.broadcast_all_local_devices(batch_spins)

    # Construct batched input data for network
    data = networks.NetworkInput(
        positions = batch_pos, 
        spins = batch_spins, 
        atoms = batch_atoms, 
        charges = batch_charges
        )

    # Initialization of network and parameters
    network = networks.Psiformer(
            nspins = nspins,
            charges = charges,
            num_dets = cfg.network.determinants,
            num_layers = cfg.network.psiformer.num_layers,
            dims_mlp_hidden = cfg.network.psiformer.mlp_hidden_dims,
            num_heads = cfg.network.psiformer.num_heads,
            dim_heads = cfg.network.psiformer.heads_dim,
            envelope = envelopes.IsotropicEnvelope,
            jastrow = jastrows.Jastrow_ee,
            ndim = cfg.system.ndim,
            act_fn = jax.nn.tanh,
            rescale_inputs = cfg.network.rescale_inputs,
            use_res = True,
            use_LN = cfg.network.psiformer.use_layer_norm,
            use_gate = True,
            separate_spin_channels = cfg.network.psiformer.separate_spin_channels, 
            orbital_bias = cfg.network.bias_orbitals
            )
    key, subkey = jax.random.split(key)
    params = network.init(
        subkey, 
        data.positions[0][0], 
        data.spins[0][0], 
        data.atoms[0][0], 
        data.charges[0][0]
        )
    del subkey

    param_sizes = jax.tree_util.tree_map(
        lambda x: x.size, params['params'])
    num_params = sum(jax.tree_util.tree_leaves(param_sizes))
    logger.info(f"Number of parameters: {num_params}")
    params = kfac_jax.utils.replicate_all_local_devices(params)

    ############################################################ 
    #                                                          #
    #                      Make Functions                      #
    #                                                          #
    ############################################################
    # Vmapped function for forward propagation of network
    signed_network_with_aux = network.apply
    logabs_network = lambda *args, **kwargs: signed_network_with_aux(*args, **kwargs)[1]
    batch_network = jax.vmap(
        logabs_network, 
        in_axes=(None, 0, 0, 0, 0), 
        out_axes=0
        )
    batch_det_weights = jax.vmap(
            lambda *args, **kwargs: signed_network_with_aux(*args, **kwargs)[2].weights, 
            in_axes = (None, 0, 0, 0, 0), 
            out_axes = 0
            )

    def network_orbitals(model, *args, **kwargs):
        return model.get_orbitals(*args, **kwargs)

    batch_orbitals = jax.vmap(
        nn.apply(network_orbitals, network), 
        in_axes = (None, 0, 0, 0, 0), 
        out_axes = 0
        )

    # Construct MCMC step and loss function
    mcmc_step = mcmc.make_mcmc_step(
        logabs_network = logabs_network,
        batch_size = device_batch_size,
        nsteps = cfg.mcmc.steps,
        proposal = cfg.mcmc.proposal,
        max_norm = cfg.mcmc.max_norm
        )

    local_energy_fn = local_energy.make_local_energy(
        logabs_network = logabs_network,
        charges = charges,
        laplacian = cfg.optim.laplacian
        )

    loss_fn = qmc_loss.make_kfac_loss(
        batch_network = batch_network,
        batch_det_weights = batch_det_weights,
        local_energy = local_energy_fn,
        clip_local_energy = cfg.optim.clip_local_energy,
        center_at_clip = cfg.optim.center_at_clip, 
        reg_weight = cfg.optim.reg_weight
        )

    ############################################################ 
    #                                                          #
    #                      Initialization                      #
    #                                                          #
    ############################################################
    # Replicate random key to all devices
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    iteration = 0  # Number of completed training steps
    adapt_frequency = cfg.mcmc.adapt_frequency
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        cfg.mcmc.move_width)
    pmoves = np.zeros(adapt_frequency)
    beta = 0.9  # EWMA weight

    # Compute the learning rate
    def lr_schedule(t: jax.Array) -> jax.Array:
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t / cfg.optim.lr.delay))), cfg.optim.lr.decay)
    # Initialization of training step and optimizer state
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    train_step, opt_state = optimizer.init_optim(
        cfg_optim = cfg.optim,
        lr_schedule = lr_schedule,
        mcmc_step = mcmc_step,
        loss_fn = loss_fn,
        params = params,
        data = data,
        sharded_key = subkeys
        )
    del subkeys
    ############################################################ 
    #                                                          #
    #                    Checkpoint Settings                   #
    #                                                          #
    ############################################################
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ckp_example = checkpoint.CheckpointData(
        iteration = iteration,
        params = params,
        data = data,
        opt_state = opt_state,
        key = subkeys,
        mcmc_width = mcmc_width,
        loss_ewma = 0.0,
        pmoves = pmoves
        )
    del subkeys

    # Restore datas from checkpoint file
    ckp_data = checkpoint.restore_from_ckp(
        restore_path = cfg.log.restore_path,
        logger = logger,
        ckp_example = ckp_example
        )
    params = ckp_data.params # Only parameter required for inference
    if cfg.mode == 'training':
        iteration = ckp_data.iteration
        data = ckp_data.data
        opt_state = ckp_data.opt_state
        key = ckp_data.key
        mcmc_width = ckp_data.mcmc_width
        loss_ewma = ckp_data.loss_ewma
        pmoves = ckp_data.pmoves

    # Create save manager
    if cfg.mode == 'training':  # Only save checkpoints for training
        save_interval_steps = cfg.log.save_interval_steps
        save_mngr, save_path = checkpoint.create_save_mngr(
            save_path = cfg.log.save_path,
            save_interval_steps = save_interval_steps,
            max_to_keep = cfg.log.max_to_keep,
            logger = logger
            )

    if iteration == 0:  # A new training/infernce run
        if cfg.mode == 'training':  # A new training run
            logger.info(f"Training mode: will start {cfg.optim.iterations} training "
                        f"steps with optimizer: {cfg.optim.optimizer}")
            if cfg.pretrain.iterations > 0:
                ############################################################ 
                #                                                          #
                #                  Hartree-Fock Pretraining                #
                #                                                          #
                ############################################################
                logger.info(f"Start Hartree-Fock pretraining ({cfg.pretrain.iterations} steps):")
                hartree_fock = pretrain.get_hf(
                    molecule = cfg.system.molecule,
                    nspins = nspins,
                    basis = cfg.pretrain.basis,
                    pyscf_mol = cfg.system.get('pyscf_mol'),
                    restricted = False
                    )  # Get Hartree-Fock results
                sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
                params, data = pretrain.pretrain_hartree_fock(
                    params = params,
                    data = data,
                    batch_network = batch_network,
                    batch_orbitals = batch_orbitals,
                    sharded_key = subkeys,
                    nspins = nspins,
                    scf_approx = hartree_fock,
                    iterations = cfg.pretrain.iterations,
                    batch_size = device_batch_size,
                    scf_fraction = cfg.pretrain.scf_fraction,
                    mcmc_width = 0.02,
                    logger = logger
                    )
                del subkeys
                logger.info("Hartree-Fock pretraining completed.")
        ############################################################ 
        #                                                          #
        #                       MCMC Burn-in                       #         
        #                                                          #
        ############################################################
        if cfg.mcmc.burn_in > 0:
            logger.info(f"Start MCMC burn-in ({cfg.mcmc.burn_in} steps):")
            p_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
            for i in range(1, cfg.mcmc.burn_in + 1):
                sharded_key, mcmc_keys = kfac_jax.utils.p_split(sharded_key)
                data, pmove = p_mcmc_step(params, data, mcmc_keys, mcmc_width)
                del mcmc_keys
                if i % 10 == 0:
                    logger.info(f'{i}: pmove = {pmove[0].item():.4f}')
            logger.info("MCMC burn-in completed.")

    if cfg.mode == 'training':  # Training mode
        ############################################################ 
        #                                                          #
        #                      Training Loop                       #
        #                                                          #
        ############################################################
        for i in range(iteration + 1, cfg.optim.iterations + 1):
            # MCMC sampling and parameter optimization
            params, data, opt_state, sharded_key, loss, aux, pmove = train_step(
                params, data, opt_state, sharded_key, mcmc_width)  # Type: optimizer.StepResults

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
            logger.info(
                f"Step {i}: E = {loss:.6f}, E_ewma = {loss_ewma:.6f}, pmove = {pmove:.4f}")
            # Checkpoint saving
            if i == 1 or i % save_interval_steps == 0:
                # Replicate data on all devices to the local device
                params_local = jax.device_get(params)
                data_local = jax.device_get(data)
                opt_state_local = jax.device_get(opt_state)
                sharded_key_local = jax.device_get(sharded_key)
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
            if i % 50 == 0: 
                logger.info(f"jastrow = {params['params']['PsiformerOrbitals']['jastrow_fn']}")
            if aux_data.weights != None:
                logger.info(f"Average det weights: {aux_data.weights}")
            '''
        save_mngr.wait_until_finished()
        logger.info(f"{cfg.optim.iterations} training steps completed. All checkpoints saved in {save_path}.")

    elif cfg.mode == 'inference':  # Inference mode
        ############################################################ 
        #                                                          #
        #                      Inference Loop                      #
        #                                                          #
        ############################################################
        logger.info("Inference mode: no checkpoint will be saved.")
        infer_step = optimizer.make_infer_step(mcmc_step, loss_fn)
        infer_step = functools.partial(infer_step, params)
        for i in range(1, cfg.optim.iterations + 1):
            data, sharded_key, loss, aux, pmove = infer_step(
                data, sharded_key, mcmc_width)
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
            logger.info(
                f"Step {i}: E = {loss:.6f}, E_ewma = {loss_ewma:.6f}, pmove = {pmove:.4f}")
        logger.info(f"{cfg.optim.iterations} inference steps completed.")
    
    else:
        raise NotImplementedError(f"Unknown mode: {cfg.mode}.")

