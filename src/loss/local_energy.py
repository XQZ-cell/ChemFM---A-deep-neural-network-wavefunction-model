# Standard library imports
from typing import Tuple, Callable, Dict

# Third-party library imports
import jax
from jax import numpy as jnp
from folx import forward_laplacian

# Local module imports
from src.networks import networks


def local_kinetic_ene(
        logabs_network: Callable,
        params: Dict,
        data: networks.NetworkInput,
        laplacian: str
        ) -> jax.Array:
    """Calculate local kinetic energy.
    
    Args:
        logabs_network: network which returns the logabs of wavefunction, a pure function.
        params: parameters of network.
        data: input of network.
        laplacian: calculation method of Laplacian operator of wavefunction. Currently only 'folx' is available.

    Returns:
        Local kinetic energy.
    """
    if laplacian == 'folx':
        def logabs(x):
            return logabs_network(params, x, data.spins, data.atoms, data.charges)
        
        fwd_f = forward_laplacian(logabs, sparsity_threshold=6)
        results = fwd_f(data.positions)
        kinetic_ene = - (results.laplacian +
                         jnp.sum(results.jacobian.dense_array ** 2)) / 2
    else:
        raise ValueError(f"Laplacian method '{laplacian}' not implemented.")
    
    return kinetic_ene


def local_potential_ene(
        charges: jax.Array,
        r_ee: jax.Array,
        r_ae: jax.Array,
        r_aa: jax.Array
        ) -> jax.Array:
    '''Calculate the local potential energy of given electronic and atomic configurations.
    
    Args:
        r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
        r_ae: atom-electron distance. Shape (nelectron, natom, 1).
        r_aa: atom-atom distance. Shape (natom, natom).
    
    Returns:
        Local potential energy.
    '''
    # electron-electron
    potential_e_e = jnp.sum(jnp.triu(1.0 / r_ee[..., 0], 1))
    # electron-nuclear
    potential_e_a = - jnp.sum(charges / r_ae[..., 0])
    # nuclear-nuclear
    charge_mat = charges[None, ...] * charges[..., None]
    potential_a_a = jnp.sum(jnp.triu(charge_mat / r_aa, 1))

    return potential_e_e + potential_e_a + potential_a_a 


def make_local_energy(
        logabs_network: Callable,
        charges: jax.Array,
        laplacian: str,
        ndim: int = 3
        ) -> Callable[[Dict, networks.NetworkInput], jax.Array]:
    """Construct local energy function.
    
    Args:
        logabs_network: network which returns the logabs of wavefunction, a pure jax function.
        charges: nuclear charges of atoms. Shape (natom,).
        laplacian: calculation method of Laplacian operator of wavefunction.
        ndim: dimensions of system.

    Returns:
        A function calculating local energy.
    """
    def local_energy(
            params: Dict, 
            data: networks.NetworkInput
            ) -> jax.Array:
        """Calculate local energy of given network parameters and inputs.
        Note that this implementation includes the nuclear-nuclear repulsive energy,
        which is computed each time based on atomic positions. For systems with fixed
        atomic positions, precomputing this term can improve performance.

        Args:
            params: parameters of network.
            data: input data of network (including atomic positions).

        Returns:
            Local energy (sum of kinetic, electron-electron, electron-nuclear, and nuclear-nuclear energies).        
        """
        # Kinetic energy
        kinetic_ene = local_kinetic_ene(
            logabs_network,
            params,
            data,
            laplacian
            )
        # Potential energy
        _, _, r_ae, r_ee = networks.create_inputs(data.positions, data.atoms, ndim = ndim)
        r_aa = jnp.linalg.norm(data.atoms[None, ...] - data.atoms[:, None], axis=-1)
        potential_ene = local_potential_ene(charges, r_ee, r_ae, r_aa)

        return kinetic_ene + potential_ene
    
    return local_energy
    