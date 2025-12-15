import jax, chex
from jax import numpy as jnp
import haiku as hk

from typing import Tuple, Callable, Sequence, Optional


# charges: atomic charges, shape (natoms,).
# atoms: atomic positions, shape (natoms, ndim).


def make_nucleus_features(
        atoms: jax.Array,
        charges: jax.Array,
        ndim: int
        ) -> jax.Array:
    """Construct nucleus and bond features to the network 
       from raw atomic charges and positions.

    Args:
        atoms: atomic positions, shape (natoms, ndim).
        charges: atomic charges, shape (natoms,).
        ndim: dimension of system. Change only with caution.
    
    Returns:
        nucleus_features: nucleus feature vectors, shape (natoms,ndim + 1).
    """
    if jnp.shape(charges)[0] != jnp.shape(atoms)[0]:
        raise ValueError(
            f"Number of atoms in atomic charge and position arrays should be equal, "
            f"got {jnp.shape(charges)[0]} and {jnp.shape(atoms)[0]}.")
    
    nucleus_features = jnp.concatenate((charges[..., None], atoms), axis=1)

    aa = jnp.reshape(atoms, [1, -1, ndim]) - jnp.reshape(atoms, [-1, 1, ndim])
    n = aa.shape[0]
    r_aa = (
        jnp.linalg.norm(aa + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    
    return nucleus_features, r_aa


class MolecularMPNN(hk.Module):
    def __init__(self, 
                 dim_nodes: int,
                 dim_edges: int, 
                 act_fn: Callable,
                 use_res: bool = True, 
                 name: Optional[str] = None):
        super().__init__(name = name)
        # Input linear layers
        self.input_linear_node = hk.Linear(
            dim_nodes,
            with_bias = False,
            name = "input_linear_node"
            )
        self.input_linear_edge = hk.Linear(
            dim_edges,
            with_bias = False,
            name = "input_linear_edge"
            )

    def __call__(
            self, 
            atoms: jax.Array, 
            charges: jax.Array
            ) -> jax.Array:
        nucleus_features, bond_features = make_nucleus_features(atoms, charges)
        node_features = self.input_linear_node(nucleus_features)
        edge_features = self.input_linear_edge(bond_features)

        return 