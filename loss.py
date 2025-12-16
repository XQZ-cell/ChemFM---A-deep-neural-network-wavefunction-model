import jax
import chex
from jax import numpy as jnp
from networks import NetworkInput
from typing import Dict, Callable, Tuple, Optional
from typing_extensions import Protocol
import kfac_jax, constants


@chex.dataclass
class AuxiliaryLossData:
    # Auxiliary data of loss calculation
    energy: jax.Array
    variance: jax.Array
    local_energies: jax.Array
    weights: jax.Array
    reg_term: jax.Array


class LossFunction(Protocol):
    def __call__(
            self,
            params: Dict,
            data: NetworkInput,
            ) -> Tuple[jax.Array, AuxiliaryLossData]:
        """Returns the loss and auxiliary data, with custom gradient rule.
         
        Args:
            params: network parameters.
            data: MCMC configuration to evaluate.
        """


def entropy(probs: jax.Array) -> jax.Array:
    """Calculate the entropy of a given discrete probability distribution."""
    entropy = - jnp.sum(probs)# * jnp.log(probs + 1e-10)  # For numerical stability
    return entropy

batch_entropy = jax.vmap(
    entropy, 
    in_axes = 0,
    out_axes = 0
    )

def make_kfac_loss(
        batch_network: Callable,
        batch_det_weights: Callable,
        local_energy: Callable,
        clip_local_energy: float,
        center_at_clip: bool,
        reg_weight: Optional[float]
        ) -> LossFunction:
    """Construct KFAC loss function with custom gradient rule.
    
    Args:
        batch_network: network ansatz, a pure function.
        local_energy: function calculating the local energy of given network and parameters(not batched).
        clip_local_energy:
    """
    batch_local_energy = jax.vmap(
        local_energy,
        in_axes=(
            None, 
            NetworkInput(
                positions = 0, 
                spins = 0, 
                atoms = 0, 
                charges = 0)
            ),
        out_axes = 0
        )

    @jax.custom_jvp
    def total_energy(
            params: Dict,
            data: NetworkInput
            ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        local_energies = batch_local_energy(params, data)
        # Batch averaged local energy (loss)
        energy = jnp.mean(local_energies)
        variance = jnp.var(local_energies)
        # Device average
        energy = constants.pmean(energy)
        
        return energy, AuxiliaryLossData(
            energy = energy, 
            local_energies = local_energies,
            variance = variance,  # This is not right for pmap!
            weights = None,
            reg_term = None
            )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        """Custom Jacobian-vector product"""
        params, data = primals
        param_tangents, data_tangents = tangents
        
        # Original loss and auxiliary data
        energy, aux_data = total_energy(params, data)
        
        # Energy cliping
        if clip_local_energy > 0.0:
            mean_energy = jnp.mean(aux_data.local_energies)
            median_energy = jnp.median(aux_data.local_energies)
            mad = jnp.mean(jnp.abs(aux_data.local_energies - median_energy))  # mean absolute derivation
            lower = median_energy - clip_local_energy * mad
            upper = median_energy + clip_local_energy * mad
            clipped_energy = jnp.clip(aux_data.local_energies, lower, upper)
            if center_at_clip:
                clipped_mean_energy = jnp.mean(clipped_energy)
                diff = clipped_energy - clipped_mean_energy
            else:
                diff = clipped_energy - mean_energy
        else:
            mean_energy = jnp.mean(aux_data.local_energies)
            clipped_energy = aux_data.local_energies
            diff = aux_data.local_energies - mean_energy

        network_primals = (
            params, 
            data.positions, 
            data.spins, 
            data.atoms, 
            data.charges
            )
        network_tangents = (
            param_tangents,
            data_tangents.positions,
            data_tangents.spins, 
            data_tangents.atoms,
            data_tangents.charges
            )
        
        # Output and tangent of network
        psi_primal, psi_tangent = jax.jvp(batch_network, network_primals, network_tangents)
        
        # Loss registration
        kfac_jax.register_normal_predictive_distribution(psi_primal[:, None]) # complex_output = False
        
        device_batch_size = jnp.shape(aux_data.local_energies)[0]
        # Calculate directional derivative of loss function w.r.t network parameters
        tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
        primals_out = (energy, aux_data)
        
        return primals_out, tangents_out

    def loss_fn(
            params: Dict,
            data: NetworkInput
            ) -> Tuple[jax.Array, AuxiliaryLossData]: 
        energy, aux_data = total_energy(params, data)
        if reg_weight:
            batch_weights = batch_det_weights(
                params, 
                data.positions, 
                data.spins, 
                data.atoms, 
                data.charges
                )  # Note that calculation of batch_weights requires another forward propagation
                   # Shape: (batch, num_dets)
            avg_entropy = jnp.mean(batch_entropy(batch_weights))  # Shape: (1,)
            reg_term = - reg_weight * avg_entropy
            loss = energy + reg_term
            aux_data = AuxiliaryLossData(
                energy = aux_data.energy,
                variance = aux_data.variance,
                local_energies = aux_data.local_energies,
                weights = jnp.mean(batch_weights, axis=0),
                reg_term = reg_term
            )
        else:
            loss = energy
        
        return loss, aux_data

    return loss_fn
