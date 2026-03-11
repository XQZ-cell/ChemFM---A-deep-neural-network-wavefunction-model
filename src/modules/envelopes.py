# Standard library imports
from typing import Optional

# Third-party library imports
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import ones


class IsotropicEnvelope(nn.Module):
    natoms: int
    dim_out: int
    
    def setup(self):
        # Initialization of parameters
        self.sigma = self.param(
            "sigma", ones, (self.natoms, self.dim_out)
            )
        self.pi = self.param(
            "pi", ones, (self.natoms, self.dim_out)
            )
    
    def __call__(self, r_ae: jnp.ndarray) -> jax.Array:
        """Constructs orbital envelopes.
        
        Args:
            r_ae: atom-electron distance. Shape (nelectron, natom, 1).

        Returns:
            Envelopes. Shape (nelectron, dim_out).
        """
        # Construct envelopes 
        exp_term = jnp.exp(- r_ae * self.sigma)
        weighted = exp_term * self.pi
        
        return jnp.sum(weighted, axis=1)