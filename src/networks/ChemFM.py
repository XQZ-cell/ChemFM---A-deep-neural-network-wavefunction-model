# Standard library imports
from typing import Sequence, Tuple, Dict, Callable, Optional

# Third-party library imports
import jax
import chex
import e3nn_jax as e3nn
from jax import numpy as jnp
from flax import linen as nn
from e3nn_jax import flax as e3nn_flax

# Local module imports
from src.utils import utils
from src.modules import MPNN
from src.modules import network_blocks, attention


def make_nuclei_features(
        atoms: jax.Array,
        charges: jax.Array,
        # filter_dict: Dict[network_blocks.Filter, float],
        max_species: int,
        ndim: int
        ) -> Tuple[jax.Array, jax.Array]:
    """Make nuclei features with charges and relative positions"""
    na = atoms.shape[0]
    assert charges.shape[0] == na
    aa, r_aa = utils.make_real_space_vector_diff(
        atoms, atoms, ndim)
    # Nuclei features
    a_features = jax.nn.one_hot(charges, max_species)
    # Nuclei edge features
    '''
    aa_features = network_blocks.apply_multi_filters(
        r_aa, aa, filter_dict)  # Shape: (na, na, (1+ndim)*n_afilters)
    '''
    return aa, r_aa, a_features#, aa_features


def get_edge_indices(na):
    i, j = jnp.meshgrid(jnp.arange(na), jnp.arange(na), indexing='ij')
    i = i.flatten()
    j = j.flatten()
    mask = i != j
    return i[mask], j[mask]


class EleFeatureLayer(nn.Module):
    filter_dict: Dict[network_blocks.Filter, float]
    ndim: int

    def setup(self):
        self.LayerNorm = nn.LayerNorm()

    def __call__(
        self, 
        pos: jax.Array, 
        spins: jax.Array
        ) -> jax.Array:
        """Make electron features with positions and spins."""
        ne = pos.shape[0]
        pos = pos.reshape((-1, self.ndim))
        assert pos.shape[0] == spins.shape[0] == ne
        ee, r_ee = utils.make_real_space_vector_diff(
            pos, pos, self.ndim)  # Shape: (ne, ne, ndim), (ne, ne, 1)
        # Electron edge features
        ee_features = network_blocks.apply_multi_filters(
            r_ee, ee, self.filter_dict)  # Shape: (ne, ne, (1+ndim)*nfilters)
        # Sum, spins and layernorm
        e_features = jnp.sum(ee_features, axis=-2)  # Shape: (ne, (1+ndim)*nfilters)
        spins_int = ((spins + 1) // 2).astype(jnp.int32)  # -1→0, +1→1
        spin_features = jax.nn.one_hot(spins_int, 2)  # Shape: (ne, 2)
        e_features = self.LayerNorm(
            jnp.concatenate((spin_features, e_features), axis=-1))

        return e_features, r_ee


class AttentionLayer(nn.Module):
    num_heads: int
    dim_heads: int
    dim_hidden: int
    act_fn: Callable
    use_res: bool
    use_LN: bool
    use_gate: bool

    def setup(self):
        # Pre-LayerNorm
        if self.use_LN:
            self.LayerNorm_1_x = nn.LayerNorm()
            self.LayerNorm_1_y = nn.LayerNorm()
            self.LayerNorm_2 = nn.LayerNorm()

        # Attention layer
        self.attention = attention.MultiHeadAttention(
            num_heads = self.num_heads,
            key_size = self.dim_heads,
            value_size = self.dim_heads,
            dim_out = self.dim_hidden,
            with_bias = False
            )
        # Gated attention linear
        self.gate_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        # Liner layer
        self.linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = True
            )
    
    def __call__(
            self, 
            x: jax.Array, 
            y: Optional[jax.Array],  # not None for cross-attn
            edges: Optional[jax.Array]
        ) -> jax.Array:
        # Pre-LN 1
        x_norm = self.LayerNorm_1_x(x) if self.use_LN else x
        if y is not None:
            y_norm = self.LayerNorm_1_y(y) if self.use_LN else y
        else:
            assert edges.shape[0] == edges.shape[1]
            y_norm = x_norm
        # Attention output. If edges is not None, attention scores will be biased.
        attn_out = self.attention(x_norm, y_norm, y_norm, edges)
        if self.use_gate:
            # Gated attention
            gated_attn = attn_out * jax.nn.sigmoid(self.gate_linear(x_norm))
            x = gated_attn + x if self.use_res else gated_attn
        else:
            x = attn_out + x if self.use_res else attn_out
        # Pre-LN 2
        x_norm = self.LayerNorm_2(x) if self.use_LN else x
        # MLP
        mlp_hidden = self.act_fn(self.linear(x_norm)) # Output MLP
        x = mlp_hidden + x if self.use_res else mlp_hidden
        
        return x


class NucleiMPNN(nn.Module):
    max_species: int
    num_layers: int
    num_mlp_layers: int
    dim_mlp: int
    max_ell: int
    num_output_irreps: int
    radial_basis: Callable[[jax.Array], jax.Array]
    n_radial_basis: int
    act_fn: Callable[[jax.Array], jax.Array]
    ndim: int
    # Nuclei MPNN
    def setup(self):
        self.input_linear_a = nn.Dense(
            features = self.num_output_irreps,
            use_bias = True
            )
        
        self.MPNNLayers = [
            MPNN.E3MPNNLayer(
                max_ell = self.max_ell,
                output_irreps = self.num_output_irreps * e3nn.Irreps("0e + 1o + 2e"),
                num_mlp_layers = self.num_mlp_layers,
                dim_mlp = self.dim_mlp,
                radial_basis = self.radial_basis,
                n_radial_basis = self.n_radial_basis
                ) for _ in range(self.num_layers)
            ]
    
    def __call__(
            self, 
            atoms: jax.Array, 
            charges: jax.Array
            ) -> jax.Array:
        output_irreps = self.num_output_irreps * e3nn.Irreps("0e + 1o + 2e")
        na = atoms.shape[0]
        assert charges.shape[0] == na

        senders, receivers = get_edge_indices(na)

        # Make nuclei features (including edge)
        aa, r_aa, a_features = make_nuclei_features(
            atoms, charges, self.max_species, self.ndim)
        a_features = self.act_fn(self.input_linear_a(a_features))  # 1-layer

        aa = e3nn.IrrepsArray("1o", aa[senders, receivers])  # (na*(na-1), ndim)
        r_aa = r_aa[senders, receivers]  # remove 0-distance, (na*(na-1), 1)

        len_padding = output_irreps.dim - self.num_output_irreps
        if len_padding < 0:
            raise ValueError(
                f"output_irreps.dim ({output_irreps.dim}) is smaller "
                "than num_output_irreps ({self.num_output_irreps}), cannot pad.")
        a_features = jnp.pad(
            a_features, [(0, 0), (0, len_padding)] , mode='constant', constant_values=0.0)  # padding 0
        a_features = e3nn.IrrepsArray(output_irreps, a_features)
        
        for i in range(self.num_layers):
            a_features = self.MPNNLayers[i](
                a_features, aa, r_aa, senders, receivers)
        
        return a_features, aa, r_aa, senders, receivers


class ChemFMOrbitals(nn.Module):
    nspins: Tuple[int, int]
    charges: jax.Array
    num_dets: int
    ########## Feature Layer ##########
    e_filter_dict: Dict[network_blocks.Filter, float]
    a_filter_dict: Dict[network_blocks.Filter, float]
    dim_e_init: int
    max_species: int
    ########### Nuclei MPNN ###########
    num_layers_a: int
    dim_hidden_a: int
    use_self_edge: bool
    #### Electron-nuclei cross-attn####
    dim_hidden_ae: int
    num_heads_ae: int
    dim_heads_ae: int
    ###### Electron transfomrer #######
    num_layers_e: int
    dim_hidden_e: int
    num_heads_e: int
    dim_heads_e: int
    ###################################
    envelope: nn.Module
    jastrow: str
    ndim: int
    act_fn: Callable
    rescale_inputs: bool
    use_res: bool
    use_LN: bool
    use_gate: bool
    use_edge_bias: bool
    separate_spin_channels: bool
    orbital_bias: bool

    def setup(self):
        N, natoms = sum(self.nspins), self.charges.shape[0]
        num_orbs_per_e = N * self.num_dets

        # Electron feature layer
        self.feat_layer_e = EleFeatureLayer(
            filter_dict = self.e_filter_dict,
            ndim = self.ndim
            )
        
        self.input_linear_e_1 = nn.Dense(
            features = self.dim_e_init,
            use_bias = True
            )
        
        self.input_linear_e_2 = nn.Dense(
            features = self.dim_hidden_e,
            use_bias = True
            )

        # Nuclei MPNN
        self.input_linear_a = nn.Dense(
            features = self.dim_hidden_a,
            use_bias = True
            )

        self.NucleiMPNNLayers = [
            MPNN.MPNNLayer(
                dim_hidden = self.dim_hidden_a,
                act_fn = self.act_fn,
                use_res = self.use_res,
                use_LN = self.use_LN,
                use_self_edge = self.use_self_edge
                ) for _ in range(self.num_layers_a)
            ]
        
        # Electron-nuclei cross-attention layer

        self.EleNucleiCrossAttnLayer = AttentionLayer(
            num_heads = self.num_heads_ae,
            dim_heads = self.dim_heads_ae,
            dim_hidden = self.dim_hidden_ae,
            act_fn = self.act_fn,
            use_res = self.use_res,
            use_LN = self.use_LN,
            use_gate = self.use_gate
            )

        # Elctron transformer layers
        self.EleTransformerLayers = [
            AttentionLayer(
                num_heads = self.num_heads_e,
                dim_heads = self.dim_heads_e,
                dim_hidden = self.dim_hidden_e,
                act_fn = self.act_fn,
                use_res = self.use_res,
                use_LN = self.use_LN,
                use_gate = self.use_gate
                ) for _ in range(self.num_layers_e)
            ]

        # Orbital projection layer
        if self.separate_spin_channels:
            self.orbital_proj_alpha = nn.Dense(
                features = num_orbs_per_e,
                use_bias = self.orbital_bias
                )
            self.orbital_proj_beta = nn.Dense(
                features = num_orbs_per_e,
                use_bias = self.orbital_bias
                )
        else:
            self.orbital_proj = nn.Dense(
                features = num_orbs_per_e,
                use_bias = self.orbital_bias
                )
        
        # Envelope and Jastrow
        self.envelope_fn = self.envelope(natoms, num_orbs_per_e)
        if self.jastrow == 'simple':
            ee_cusp_fun = jastrows.simple_ee_cusp_fun
        elif self.jastrow == 'exp':
            ee_cusp_fun = jastrows.exp_ee_cusp_fun
        else:
            raise NotImplementedError(f"Unknown Jastrow type: {self.jastrow}.")
        self.jastrow_fn = jastrows.Jastrow_ee(ee_cusp_fun)
    
    def __call__(
            self, 
            pos: jax.Array, 
            spins: jax.Array, 
            atoms: jax.Array, 
            charges: jax.Array
        ) -> jax.Array:
        na = atoms.shape[0]
        assert charges.shape[0] == na
        # Make electron features
        e_features, r_ee = self.feat_layer_e(pos, spins)
        e_features = self.act_fn(self.input_linear_e_1(e_features))  # 1-layer MLP
        # Make nuclei features (including edge)
        a_features, aa_features = make_nuclei_features(
            atoms, charges, self.a_filter_dict, self.max_species, self.ndim)
        a_features = self.act_fn(self.input_linear_a(a_features))  # 1-layer MLP
        # Update nuclei features using MPNN
        for NucleiMPNNLayer in self.NucleiMPNNLayers:
            a_features, aa_features = NucleiMPNNLayer(a_features, aa_features)
        # Elctron and nuclei feature fusion
        _, r_ae = utils.make_real_space_vector_diff(
            atoms, pos, self.ndim)
        e_features = self.EleNucleiCrossAttnLayer(
            e_features, a_features, r_ae.squeeze(-1))
        e_features = self.act_fn(self.input_linear_e_2(e_features))  # 1-layer MLP
        # Electron transformer
        for EleTransformerLayer in self.EleTransformerLayers:
            if self.use_edge_bias:
                e_features = EleTransformerLayer(e_features, None, r_ee.squeeze(-1))
            else:
                e_features = EleTransformerLayer(e_features, None, None)
        
        # Orbital generator
        # Orbital projection
        if self.separate_spin_channels:
            e_features_alpha = e_features[:self.nspins[0]]
            e_features_beta = e_features[self.nspins[0]:]
            orbitals_alpha = self.orbital_proj_alpha(e_features_alpha)
            orbitals_beta = self.orbital_proj_beta(e_features_beta)
            orbitals = jnp.concatenate([orbitals_alpha, orbitals_beta], axis=0)
        else:
            orbitals = self.orbital_proj(e_features)
        
        # Apply envelope
        orbitals = orbitals * self.envelope_fn(r_ae)
        
        # Reshape and transpose to get [num_dets, nelectrons, norbitals]
        det_shape = (orbitals.shape[0], orbitals.shape[1] // self.num_dets)
        orbitals = orbitals.reshape(det_shape[0], self.num_dets, det_shape[1]).transpose(1, 0, 2)
        
        # Apply Jastrow
        jastrow = jnp.exp(
            self.jastrow_fn(self.nspins, r_ee) / sum(self.nspins))
        orbitals = orbitals * jastrow

        return orbitals