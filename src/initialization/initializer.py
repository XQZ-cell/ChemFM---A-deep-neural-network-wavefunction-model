# Standard library imports
import functools
from typing import Mapping, Sequence, Tuple, Callable, Any

# Third-party library imports
import jax
import chex
import kfac_jax
import numpy as np
import ml_collections
from absl import logging
from jax import numpy as jnp
from flax import linen as nn

# Local module imports
from src import mcmc, optimizer, constants
from src.utils import system
from src.networks import networks
from src.loss import local_energy
from src.loss import loss as qmc_loss


def setup_devices_and_batch(
        total_batch_size: int
        ) -> Tuple[int, int]:
    """Get device information and calculate batch size."""
    # Show device information
    devices = jax.devices()
    num_devices = len(devices)
    logging.info(f"Found {num_devices} available JAX devices:")
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
        logging.info(info_str)
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
        logging.warning(
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



class QMCInitializer:
    def __init__(
            self, 
            cfg: ml_collections.ConfigDict,
            key: chex.PRNGKey
        ):
        self.cfg = cfg
        self.key = key
        # Device info
        self.leading_data_shape = setup_devices_and_batch(cfg.batch_size)
        self.device_batch_size = self.leading_data_shape[1]
        # Chemical system
        self.nspins = cfg.system.electrons # A Tuple of (nalpha, nbeta)
        self.charges = jnp.array([atom.charge for atom in cfg.system.molecule])  # Shape: (natoms,)

    def init_data(self) -> networks.NetworkInput:

        # Get system information
        atoms = jnp.stack([jnp.array(atom.coords) for atom in self.cfg.system.molecule])  # Shape: (natoms, ndim)

        # Generate batched atomic configurations
        batch_atoms = jnp.tile(atoms[None, ...], [self.device_batch_size, 1, 1])
        batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)  # Shape: (num_devices, device_batch_size, natoms, ndim)
        batch_charges = jnp.tile(self.charges[None, ...], [self.device_batch_size, 1])
        batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)  # Shape: (num_devices, device_batch_size, ne=nalpha+nbeta)

        # Initialization of electron configurations
        _, subkey = jax.random.split(self.key)
        batch_pos, batch_spins = init_electrons(
            subkey,
            molecule = self.cfg.system.molecule,
            electrons = self.nspins,
            batch_size = self.cfg.batch_size,
            init_width = self.cfg.mcmc.init_width,
            core_electrons = {},
            max_iter = 10_000
            )
        del subkey

        batch_pos = batch_pos.reshape(self.leading_data_shape + (-1,))  # Shape: (num_devices, device_batch_size, ne*ndim)
        batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
        batch_spins = batch_spins.reshape(self.leading_data_shape + (-1,))  # Shape: (num_devices, device_batch_size, ne)
        batch_spins = kfac_jax.utils.broadcast_all_local_devices(batch_spins)

        # Construct batched input data for network
        data = networks.NetworkInput(
            positions = batch_pos, 
            spins = batch_spins, 
            atoms = batch_atoms, 
            charges = batch_charges
            )
        
        return data
    
    def init_model(
            self, 
            data: networks.NetworkInput
        ) -> Tuple[nn.Module, Any]:
        # Initialization of network
        if self.cfg.network.network_type == 'psiformer':
            network = networks.Psiformer(
                nspins = self.nspins,
                charges = self.charges,
                num_dets = self.cfg.network.determinants,
                num_layers = self.cfg.network.psiformer.num_layers,
                dims_mlp_hidden = self.cfg.network.psiformer.mlp_hidden_dims,
                num_heads = self.cfg.network.psiformer.num_heads,
                dim_heads = self.cfg.network.psiformer.heads_dim,
                envelope = self.cfg.network.envelope,
                jastrow = self.cfg.network.jastrow,
                ndim = self.cfg.system.ndim,
                act_fn = self.cfg.network.activation_fun,
                rescale_inputs = self.cfg.network.rescale_inputs,
                use_res = True,
                use_LN = self.cfg.network.psiformer.use_layer_norm,
                use_gate = self.cfg.network.psiformer.use_gate,
                use_edge_bias = self.cfg.network.psiformer.use_edge_bias,
                separate_spin_channels = self.cfg.network.psiformer.separate_spin_channels, 
                orbital_bias = self.cfg.network.bias_orbitals
                )
        
        else:
            raise NotImplementedError(f"Unknown model name: {self.cfg.network.network_type}.")

        # Initialization of parameters
        _, subkey = jax.random.split(self.key)
        params = network.init(
            subkey, 
            data.positions[0][0], 
            data.spins[0][0], 
            data.atoms[0][0], 
            data.charges[0][0]
            )
        del subkey
        # Replicate parameters to all devices
        params = kfac_jax.utils.replicate_all_local_devices(params)

        return network, params
    
    def get_param_size(
            self, 
            params: Any
            ) -> int:
        param_sizes = jax.tree_util.tree_map(
            lambda x: x.size, params['params'])
        num_params = sum(jax.tree_util.tree_leaves(param_sizes)) // self.leading_data_shape[0]
        
        return num_params
    
    def make_functions(
            self,
            network: nn.Module,
            ) -> Tuple[Callable, Callable, Callable, Callable]:
        """Function factory."""
        signed_network_with_aux = network.apply
        logabs_network = lambda *args, **kwargs: signed_network_with_aux(*args, **kwargs)[1]
        batch_network = jax.vmap(
            logabs_network, 
            in_axes = (None, 0, 0, 0, 0), 
            out_axes = 0
            )

         # Function for intermediate values
        def network_inter_value(params, pos, spins, atoms, charges):
            return network.apply(params, pos, spins, atoms, charges, mutable='intermediates')[1]
        get_inter_value = constants.pmap(
            jax.vmap(
                network_inter_value, 
                in_axes = (None, 0, 0, 0, 0), 
                out_axes = 0
                )
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
            batch_size = self.device_batch_size,
            nsteps = self.cfg.mcmc.steps,
            proposal = self.cfg.mcmc.proposal,
            sampler_params = self.cfg.mcmc.sampler_params
            )

        local_energy_fn = local_energy.make_local_energy(
            logabs_network = logabs_network,
            charges = self.charges,
            laplacian = self.cfg.optim.laplacian
            )

        loss_fn = qmc_loss.make_kfac_loss(
            batch_network = batch_network,
            batch_det_weights = batch_det_weights,
            local_energy = local_energy_fn,
            clip_local_energy = self.cfg.optim.clip_local_energy,
            center_at_clip = self.cfg.optim.center_at_clip, 
            reg_weight = self.cfg.optim.reg_weight
            )

        return batch_network, batch_orbitals, batch_det_weights, get_inter_value, mcmc_step, loss_fn
    
    def init_optimizer(
            self,
            params: Any,
            data: networks.NetworkInput,
            mcmc_step: Callable,
            loss_fn: Callable
            ) -> Tuple[optimizer.OptStep, optimizer.OptimizerState, chex.PRNGKey]:
        # Replicate random key to all devices
        sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(self.key)

        # Learning rate schedule
        def lr_schedule(t: jax.Array) -> jax.Array:
            return self.cfg.optim.lr.rate * jnp.power(
                (1.0 / (1.0 + (t / self.cfg.optim.lr.delay))), 
                self.cfg.optim.lr.decay
                )
        # Initialization of optimizer
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        train_step, opt_state = optimizer.init_optim(
            cfg_optim = self.cfg.optim,
            lr_schedule = lr_schedule,
            mcmc_step = mcmc_step,
            loss_fn = loss_fn,
            params = params,
            data = data,
            sharded_key = subkeys
            )
        del subkeys

        #return train_step, opt_state, sharded_key

        if self.cfg.mode == 'inference' or self.cfg.mode == 'analysis':
            train_step = optimizer.make_infer_step(mcmc_step, loss_fn)
        
        return train_step, opt_state, sharded_key