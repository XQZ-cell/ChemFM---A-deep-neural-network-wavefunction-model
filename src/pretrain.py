# Standard library imports
from typing import Dict, Callable, Mapping, Sequence, Tuple, Optional

# Third-party library imports
import jax
import chex
import kfac_jax
import optax
import pyscf
import numpy as np
from jax import numpy as jnp

# Local module imports
from src import constants
from src import mcmc
from src.networks import networks
from src.utils import system, scf


def get_hf(
        molecule: Sequence[system.Atom] | None = None,
        nspins: Tuple[int, int] | None = None,
        basis: str | None = 'sto-3g',
        pyscf_mol: pyscf.gto.Mole | None = None,
        restricted: bool | None = False
        ) -> scf.Scf:
    if pyscf_mol:
        scf_approx = scf.Scf(
            pyscf_mol=pyscf_mol,
            restricted=restricted)
    else:
        scf_approx = scf.Scf(
            molecule,
            nelectrons=nspins,
            basis=basis,
            ecp=None,
            core_electrons=None,
            restricted=restricted)
    scf_approx.run(
        excitations=0, 
        excitation_type='ordered')
    
    return scf_approx

def make_pretrain_loss_fn(
        nspins: Tuple[int, int], 
        batch_orbitals: Callable,
        scf_approx: scf.Scf
        ) -> Callable:
    def loss_fn(
            params: Dict, 
            data: networks.NetworkInput
            ) -> jax.Array:
        target = scf_approx.eval_orbitals(data.positions, nspins)  # Shape: (batch, nspins, na/nb, na/nb)
        orbitals = batch_orbitals(
            params, data.positions, data.spins, data.atoms, data.charges)  # Shape: (batch, ndets, na+nb, na+nb)
        dims = target[0].shape[:-2]  # leading dimensions
        na = target[0].shape[-2]
        nb = target[1].shape[-2]
        target = jnp.concatenate(
            (
                jnp.concatenate(
                    (target[0], jnp.zeros(dims + (na, nb))), axis=-1),
                jnp.concatenate(
                    (jnp.zeros(dims + (nb, na)), target[1]), axis=-1),
            ),
            axis=-2,
        )  # Shape: (batch, na+nb, na+nb)
        return jnp.mean((target[:, None, ...] - orbitals)**2)  # Mean Squared Error
    
    return loss_fn


def make_pretrain_step(
        nspins: Tuple[int, int],
        batch_size: int,
        batch_network: Callable,
        batch_orbitals: Callable,
        scf_fraction: float,
        scf_approx: scf.Scf,
        optimizer_update: optax.TransformUpdateFn,
        ) -> Callable:
    # Create pretraining sampling network
    scf_network = lambda fn, x: fn(x, nspins)[1]
    if scf_fraction < 1e-6:
        def mcmc_network(full_params, pos, spins, atoms, charges):
            return batch_network(full_params['network'], pos, spins, atoms, charges)
    elif scf_fraction > 0.999999:
        def mcmc_network(full_params, pos, spins, atoms, charges):
            del spins, atoms, charges
            return scf_network(full_params['scf'].eval_slater, pos)
    else:
        def mcmc_network(full_params, pos, spins, atoms, charges):
            log_network = batch_network(
                full_params['network'], pos, spins, atoms, charges)
            log_scf = scf_network(full_params['scf'].eval_slater, pos)
            return (1 - scf_fraction) * log_network + scf_fraction * log_scf
    
    # Make MCMC sampling step
    mcmc_pretrain_step = mcmc.make_pretrain_mcmc_step(
        mcmc_network, 
        batch_size = batch_size, 
        nsteps = 1
        )
    # Make PMAPPED pretraining loss function
    loss_fn = make_pretrain_loss_fn(nspins, batch_orbitals, scf_approx)
    value_and_grad = jax.value_and_grad(loss_fn, argnums=0)

    def pretrain_step(
            params: Dict,
            data: networks.NetworkInput,
            opt_state: optax.OptState,
            key: chex.PRNGKey,
            width: jax.Array
            ):
        """
        Note that all of its inputs must be replicated to all devices first.
        """
        # MCMC sampling
        full_params = {'network': params, 'scf': scf_approx}
        # Note that the value of 'scf' is a Scf object, not PyTree.
        key, mcmc_key = jax.random.split(key)
        data, pmove = mcmc_pretrain_step(
            full_params, data, mcmc_key, width)
        # Compute loss value and gradients
        loss, grads = value_and_grad(params, data)
        grads = constants.pmean(grads)  # Device average gradients
        # Update parameters
        updates, opt_state = optimizer_update(
            grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, data, opt_state, key, loss, pmove
    
    return pretrain_step


def pretrain_hartree_fock(
        params: Dict,
        data: networks.NetworkInput,
        batch_network: Callable,
        batch_orbitals: Callable,
        sharded_key: chex.PRNGKey,
        nspins: Tuple[int, int],
        scf_approx: scf.Scf,
        iterations: int,
        batch_size: int,
        scf_fraction: float,
        mcmc_width: float,
        logger: Optional[Callable]
        ) -> Tuple[Dict, networks.NetworkInput]:
    # Initialize the optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3.e-4)
        )
    opt_state = constants.pmap(optimizer.init)(params)

    # Make pretrain step function
    step = make_pretrain_step(
        nspins = nspins, 
        batch_size = batch_size,
        batch_network = batch_network, 
        batch_orbitals = batch_orbitals,   
        scf_fraction = scf_fraction,
        scf_approx = scf_approx, 
        optimizer_update = optimizer.update
        )
    step = constants.pmap(step, donate_argnums=1)
    
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width)
    adapt_frequency = 100
    pmoves = np.zeros(adapt_frequency)  # Use changeable np array
    # Pretrain iterations
    for i in range(1, iterations + 1):
        params, data, opt_state, sharded_key, loss, pmove = step(
            params, data, opt_state, sharded_key, mcmc_width)
        # Update mcmc width
        mcmc_width, pmoves = mcmc.update_mcmc_width(
            t = i,
            width = mcmc_width,
            adapt_frequency = adapt_frequency,
            pmove = pmove,
            pmoves = pmoves
            )
        if logger:
            loss = jax.device_get(loss)[0].item()
            pmove = jax.device_get(pmove)[0].item()
            logger.info(f'{i}: loss={loss:.6f}, pmove={pmove:.4f}')
    
    return params, data

