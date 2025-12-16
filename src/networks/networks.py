# Standard library imports
from typing import Tuple, Callable, Sequence, Optional

# Third-party library imports
import jax
import chex
from jax import numpy as jnp
from flax import linen as nn
#from flax.linen.initializers import lecun_normal, zeros

# Local module imports
from src.modules import attention
from src.modules import network_blocks


@chex.dataclass
class NetworkInput:
    """Data passed to network.
    
    Shapes given for an unbatched element (i.e. a single MCMC configuration).
    
    NOTE:
        the networks are written in batchless form. Typically one then maps
        (pmap+vmap) over every attribute of FermiNetData (nb this is required if
        using KFAC, as it assumes the FIM is estimated over a batch of data), but
        this is not strictly required. If some attributes are not mapped, then JAX
        simply broadcasts them to the mapped dimensions (i.e. those attributes are
        treated as identical for every MCMC configuration.

    Attributes:
        positions: walker positions, shape (nelectrons*ndim,).
        spins: spins of each walker, shape (nelectrons,).
        atoms: atomic positions, shape (natoms, ndim).
        charges: atomic charges, shape (natoms,).
    """

    # We need to be able to construct instances of this with leaf nodes as jax
    # arrays (for the actual data) and as integers (to use with in_axes for
    # jax.vmap etc). We can type the struct to be either all arrays or all ints
    # using Generic, it just slightly complicates the type annotations in a few
    # functions (i.e. requires FermiNetData[jnp.ndarray] annotation).
    positions: jax.Array
    spins: jax.Array
    atoms: jax.Array
    charges: jax.Array

"""
@chex.dataclass
class NetworkOption:
    # Hyperparameters of the network.
    num_dets: int
    num_layers: int
    dims_mlp_hidden: int
    num_heads: int
    dim_heads: int
    envelope: nn.Module
    jastrow: nn.Module
    ndim: int
    act_fn: Callable[[jax.Array], jax.Array]
    rescale_inputs: bool
    orbital_bias: bool
"""

def create_inputs(
        pos: jax.Array,
        atoms: jax.Array,
        ndim: int,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Constructs inputs to the network from raw electron and atomic positions.

    Args:
        pos: electron positions. Shape (nelectrons*ndim,).
        atoms: atom positions. Shape (natoms, ndim).
        nspins: tuple of numbers of spin-up and spin-down electrons.
                Assuming the first spins[0] electrons are alpha, and the last spins[1] are beta.
        ndim: dimension of system. Change only with caution.
    
    Returns:
        ae, ee, r_ae, r_ee tuple, where:
            ae: electron-atom position vector differences. Shape (nelectron, natom, ndim).
            ee: electron-electron position vector differences. Shape (nelectron, nelectron, ndim).
            r_ae: electron-atom distance. Shape (nelectron, natom, 1).
            r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
        The diagonal terms in r_ee are masked out such that the gradients of these
        terms are also zero.
    """
    assert atoms.shape[-1] == ndim

    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    # Avoid computing the norm of zero, as is has undefined grad
    n = ee.shape[0]
    r_ee = (
        jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

    return ae, ee, r_ae, r_ee[..., None]


def make_features(
        ae: jax.Array,
        ee: jax.Array,
        r_ae: jax.Array,
        r_ee: jax.Array,
        nspins: Optional[Tuple[int, int]],
        rescale_inputs: bool
        ) -> Tuple[jax.Array, jax.Array]:
    """Construct input features from outcomes of create_inputs().

    Args:
        ae: electron-atom position vector differences. Shape (nelectron, natom, ndim).
        ee: electron-electron position vector differences. Shape (nelectron, nelectron, ndim).
        r_ae: electron-atom distance. Shape (nelectron, natom, 1).
        r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
        nspins: numbers of electrons with different spins. A tuple of two integers.
        rescale_inputs: if true, rescale the inputs so they grow as log(|r|).
    
    Returns:
        ae_features, ee_features tuple, where:
            ae_features: electron-atom feature vector. 
                         Shape (nelectron, natom*(ndim+1)+1) if nspins else (nelectron, natom*(ndim+1)).
            ee_features: electron-electron feature vector. 
                         Shape (nelectron, nelectron, ndim+1).
    """
    if rescale_inputs:
        log_r_ae = jnp.log(1 + r_ae)  # grows as log(r) rather than r
        ae_features = jnp.concatenate((log_r_ae, ae * log_r_ae / r_ae), axis=2)
        # ln(1 + r_ae) / r_ae

        log_r_ee = jnp.log(1 + r_ee)
        ee_features = jnp.concatenate((log_r_ee, ee * log_r_ee / r_ee), axis=2)

    else:
        ae_features = jnp.concatenate((r_ae, ae), axis=2)
        ee_features = jnp.concatenate((r_ee, ee), axis=2)

    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])

    if nspins:
        assert sum(nspins) == ae_features.shape[0]
        spins_a, spins_b = jnp.ones((nspins[0], 1)), -jnp.ones((nspins[1], 1))
        spins = jnp.concatenate((spins_a, spins_b), axis=0)
        ae_features = jnp.concatenate((spins, ae_features), axis=1)

    return ae_features, ee_features

'''
def make_orb_param_initializer(
        base_init: Callable, 
        noise_scale: float, 
        num_dets: int):
    """创建基础矩阵并复制添加噪声的初始化器"""
    
    def init_fn(key, shape, dtype=jnp.float32):
        dim_in, dim_out = shape
        N = dim_out // num_dets  # Size of each det
        # Base weight
        base_key, noise_key = jax.random.split(key)
        base_weight = base_init(base_key, (dim_in, N), dtype)
        # All weights with small noise
        noise_keys = jax.random.split(noise_key, num_dets)
        weights = []
        for i in range(num_dets):
            noise = jax.random.normal(noise_keys[i], (dim_in, N), dtype) * noise_scale
            block = base_weight + noise
            weights.append(block)
        # Concatenation
        weight = jnp.concatenate(weights, axis=1)

        return weight
    
    return init_fn
'''


class PsiformerLayer(nn.Module):
    num_heads: int
    dim_heads: int
    dims_mlp_hidden: int
    act_fn: Callable
    use_res: bool
    use_LN: bool
    use_gate: bool
    
    def setup(self):
        # Pre-LayerNorm
        if self.use_LN:
            self.LayerNorm_1 = nn.LayerNorm()
            self.LayerNorm_2 = nn.LayerNorm()

        # Self-attention layer
        self.attention = attention.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.dim_heads,
            value_size=self.dim_heads,
            dim_out=self.dims_mlp_hidden,
            with_bias=False
            )
        # Gated attention linear
        self.gate_linear = nn.Dense(
            features=self.dims_mlp_hidden,
            use_bias=False
            )
        # Liner layer
        self.linear_1 = nn.Dense(
            features=self.dims_mlp_hidden,
            use_bias=True
            )
        self.linear_2 = nn.Dense(
            features=self.dims_mlp_hidden,
            use_bias=True
            )

    def __call__(self, x) -> jax.Array:
        # Pre-LN 1
        x_norm = self.LayerNorm_1(x) if self.use_LN else x
        # Attention output
        attn_out = self.attention(x_norm, x_norm, x_norm)
        if self.use_gate:
            # Gated attention
            gated_attn = attn_out * jax.nn.sigmoid(self.gate_linear(x_norm))
            x = gated_attn + x if self.use_res else gated_attn
        else:
            x = attn_out + x if self.use_res else attn_out
        # Pre-LN 2
        x_norm = self.LayerNorm_2(x) if self.use_LN else x
        # MLP
        mlp_hidden = self.linear_2(self.act_fn(self.linear_1(x_norm))) # 2 Dense and 1 activation
        x = mlp_hidden + x if self.use_res else mlp_hidden
        
        return x


class PsiformerOrbitals(nn.Module):
    nspins: Tuple[int, int]
    charges: jax.Array
    num_dets: int
    num_layers: int
    dims_mlp_hidden: int
    num_heads: int
    dim_heads: int
    envelope: nn.Module
    jastrow: nn.Module
    ndim: int
    act_fn: Callable
    rescale_inputs: bool
    use_res: bool
    use_LN: bool
    use_gate: bool
    separate_spin_channels: bool
    orbital_bias: bool

    def setup(self):
        N, natoms = sum(self.nspins), self.charges.shape[0]
        num_orbs_per_e = N * self.num_dets
        # Input linear layer
        self.input_linear = nn.Dense(
            features=self.dims_mlp_hidden,
            use_bias=False
            )
        
        # Psiformer layers
        self.psiformer_layers = [
            PsiformerLayer(
                num_heads = self.num_heads,
                dim_heads = self.dim_heads,
                dims_mlp_hidden = self.dims_mlp_hidden,
                act_fn = self.act_fn,
                use_res = self.use_res,
                use_LN = self.use_LN,
                use_gate = self.use_gate
                ) for _ in range(self.num_layers)
            ]
        
        # Orbital projection layer
        # Create custom initializer for orbital projection
        '''
        orb_kernel_init = make_orb_param_initializer(
            base_init=lecun_normal(),
            noise_scale=1e-4,
            num_dets=self.num_dets
            )
        '''
        if self.separate_spin_channels:
            self.orbital_proj_alpha = nn.Dense(
                features=num_orbs_per_e,
                #kernel_init = orb_kernel_init, 
                use_bias=self.orbital_bias
                )
            self.orbital_proj_beta = nn.Dense(
                features=num_orbs_per_e,
                #kernel_init = orb_kernel_init, 
                use_bias=self.orbital_bias
                )
        else:
            self.orbital_proj = nn.Dense(
                features=num_orbs_per_e,
                #kernel_init = orb_kernel_init, 
                use_bias=self.orbital_bias
                )
        
        # Envelope and Jastrow
        self.envelope_fn = self.envelope(natoms, num_orbs_per_e)
        self.jastrow_fn = self.jastrow()

    def __call__(self, pos, spins, atoms, charges) -> jax.Array:
        ae, ee, r_ae, r_ee = create_inputs(pos, atoms, self.ndim)
        '''
        ae_features, _ = make_features(
            ae, ee, r_ae, r_ee, self.nspins, self.rescale_inputs)  # Shape: (nelectron, dim_mlp_hidden)
        '''
        ae_features, _ = make_features(
            ae, ee, r_ae, r_ee, None, self.rescale_inputs)

        # Input linear layer
        ae_features = self.input_linear(ae_features)
        
        # Psiformer layers
        for psiformer_layer in self.psiformer_layers:
            ae_features = psiformer_layer(ae_features)
        
        # Orbital projection
        if self.separate_spin_channels:
            ae_features_alpha = ae_features[:self.nspins[0]]
            ae_features_beta = ae_features[self.nspins[0]:]
            orbitals_alpha = self.orbital_proj_alpha(ae_features_alpha)
            orbitals_beta = self.orbital_proj_beta(ae_features_beta)
            orbitals = jnp.concatenate([orbitals_alpha, orbitals_beta], axis=0)
        else:
            orbitals = self.orbital_proj(ae_features)
        
        # Apply envelope
        orbitals = orbitals * self.envelope_fn(r_ae)
        
        # Reshape and transpose to get [num_dets, nelectrons, norbitals]
        det_shape = (orbitals.shape[0], orbitals.shape[1] // self.num_dets)
        orbitals = orbitals.reshape(det_shape[0], self.num_dets, det_shape[1]).transpose(1, 0, 2)
        
        # Apply Jastrow
        jastrow = jnp.exp(
            self.jastrow_fn(self.nspins, r_ee) / sum(self.nspins)
            )
        orbitals = orbitals * jastrow

        return orbitals


class Psiformer(nn.Module):
    nspins: Tuple[int, int]
    charges: jax.Array
    num_dets: int
    num_layers: int
    dims_mlp_hidden: int
    num_heads: int
    dim_heads: int
    envelope: nn.Module
    jastrow: nn.Module
    ndim: int
    act_fn: Callable[[jax.Array], jax.Array]
    rescale_inputs: bool
    use_res: bool
    use_LN: bool
    use_gate: bool
    separate_spin_channels: bool
    orbital_bias: bool

    def setup(self):
        self.PsiformerOrbitals = PsiformerOrbitals(
            nspins=self.nspins,
            charges=self.charges,
            num_dets=self.num_dets,
            num_layers=self.num_layers,
            dims_mlp_hidden=self.dims_mlp_hidden,
            num_heads=self.num_heads,
            dim_heads=self.dim_heads,
            envelope=self.envelope,
            jastrow=self.jastrow,
            ndim=self.ndim,
            act_fn=self.act_fn,
            rescale_inputs=self.rescale_inputs,
            use_res=self.use_res, 
            use_LN=self.use_LN, 
            use_gate=self.use_gate,
            separate_spin_channels=self.separate_spin_channels,
            orbital_bias=self.orbital_bias
            )
    
    def __call__(self, pos, spins, atoms, charges) -> Tuple[jax.Array, jax.Array]:
        orbitals = self.PsiformerOrbitals(pos, spins, atoms, charges)
        result = network_blocks.logdet_matmul(orbitals)

        return result

    
    def get_orbitals(self, pos, spins, atoms, charges) -> Tuple[jax.Array, jax.Array]:
        orbitals = self.PsiformerOrbitals(pos, spins, atoms, charges)
        result = orbitals

        return result