# Standard library imports
import functools
from typing import Tuple, Callable, Sequence, Optional

# Third-party library imports
import jax
import chex
from jax import numpy as jnp
from flax import linen as nn
from typing_extensions import Protocol

# Local module imports
from src.modules.network_blocks import exp_filter as exp, sigmoid_filter as sigmoid


class MPNNLayer(nn.Module):
    dim_hidden: int
    act_fn: Callable
    use_res: bool
    use_LN: bool
    use_self_edge: bool

    def setup(self):
        # LayerNorm
        if self.use_LN:
            self.LayerNorm_1 = nn.LayerNorm()
            self.LayerNorm_2 = nn.LayerNorm()
            self.LayerNorm_3 = nn.LayerNorm()
        
        self.input_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.receiver_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.sender_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.edge_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = True
            )
        self.output_linear = nn.Dense(
            features = self.dim_hidden,  # dim_out = dim_hidden
            use_bias = True
            )

    def __call__(
            self, 
            x: jax.Array, 
            edge_embedding: jax.Array
            ) -> Tuple[jax.Array, jax.Array]:
        """Message passing neural network(MPNN) layer.

        Args:
            x: Shape: (N, dim_hidden)
            edge_embedding: Shape: (N, N*edge_features)
        
        Returns:
            Updated node features and edge_embedding. 
            Shape: (N, dim_out=dim_hidden), (N, N*dim_hidden)
        """
        # Reshape edge embedding
        N = x.shape[0]
        assert edge_embedding.shape[0] == N
        #edge_embedding = edge_embedding.reshape((N, N, -1))  # Shape: (N, N*edge_features) -> (N, N, edge_features)
        edge_embedding = self.edge_linear(edge_embedding)  # Shape: (N, N, dim_hidden)
        # Mask self-edge
        if not self.use_self_edge:
            # Make masked message
            mask = (1 - jnp.eye(N))[..., None]  # Shape: (N, N, 1)
            edge_embedding = edge_embedding * mask

        x_norm = self.LayerNorm_1(x) if self.use_LN else x
        # Input Dense
        x_norm = self.input_linear(x_norm)
        # Make features of receivers and senders
        receiver_feat = self.receiver_linear(x_norm)
        sender_feat = self.sender_linear(x_norm)
        # Make message between nodes
        message = self.act_fn(receiver_feat[:, None, :] + sender_feat[None, :, :]) * edge_embedding  # Shape: (N, N, dim_hidden)
        # Message aggregation
        messgae_sum = jnp.sum(message, axis=-2)
        messgae_norm = self.LayerNorm_2(messgae_sum) if self.use_LN else messgae_sum
        x = x + messgae_norm if self.use_res else messgae_norm
        # Output MLP
        x_norm = self.LayerNorm_3(x) if self.use_LN else x
        mlp_out = self.act_fn(self.output_linear(x_norm))
        x = x + mlp_out if self.use_res else mlp_out
        
        #edge_embedding = edge_embedding.reshape((N, -1))  # Shape: (N, N*dim_hidden)

        return x, edge_embedding


class DoubleInputMPNNLayer(nn.Module):
    dim_hidden: int
    act_fn: Callable
    use_res: bool
    use_LN: bool
    extract_message: bool
    """Message passing neural network module with separate receiver 
    and sender nodes.
    """
    def setup(self):
        # Pre-LayerNorm
        if self.use_LN:
            self.LayerNorm_x_1 = nn.LayerNorm()
            self.LayerNorm_y = nn.LayerNorm()
            self.LayerNorm_x_2 = nn.LayerNorm()
            self.LayerNorm_x_3 = nn.LayerNorm()
        
        self.input_linear_x = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.input_linear_y = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.receiver_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.sender_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.edge_linear = nn.Dense(
            features = self.dim_hidden,
            use_bias = False
            )
        self.output_linear = nn.Dense(
            features = self.dim_hidden,  # dim_out = dim_hidden
            use_bias = True
            )

    def __call__(
            self, 
            x: jax.Array, 
            y: jax.Array,
            edge_embedding: jax.Array
            ) -> jax.Array:
        """Message passing neural network with separate receiver 
        and sender nodes.

        Args:
            x: receiver features. Shape: (N, dim_hidden)
            y: sender features. Shape: (M, any)
            edge_embedding: Shap: (N, M, dim_hidden)
        
        Returns:
            Updated node features. Shape: (N, dim_out=dim_hidden)
        """
        x_norm = self.LayerNorm_x_1(x) if self.use_LN else x
        y_norm = self.LayerNorm_y(y) if self.use_LN else y
        # Input Dense
        x_norm = self.input_linear_x(x_norm)
        y_norm = self.input_linear_y(y_norm)
        # Make features of receivers and senders
        receiver_feat = self.receiver_linear(x_norm)  # Shape: (N, dim_hidden)
        sender_feat = self.sender_linear(y_norm)  # Shape: (M, dim_hidden)
        # Make message
        edge_embedding = self.edge_linear(edge_embedding)
        message = self.act_fn(receiver_feat[:, None, :] + sender_feat[None, :, :])
        message = message * edge_embedding  # Shape: (N, M, dim_hidden)
        # Message aggregation
        if self.use_res:
            x = x + jnp.sum(message, axis=-2)
        else:
            x = jnp.sum(message, axis=-2)
        x = self.LayerNorm_x_2(x) if self.use_LN else x
        # Output MLP
        x_norm = self.LayerNorm_x_3(x) if self.use_LN else x
        if self.use_res:
            x = x + self.act_fn(self.output_linear(x_norm))
        else: 
            x = self.act_fn(self.output_linear(x_norm))

        if self.extract_message:
            return x, message
        else:
            return x