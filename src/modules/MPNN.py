# Standard library imports
import functools
from typing import Tuple, Callable, Sequence, Optional

# Third-party library imports
import jax
import chex
import e3nn_jax as e3nn
from jax import numpy as jnp
from flax import linen as nn
from typing_extensions import Protocol
from e3nn_jax import flax as e3nn_flax

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


class E3MPNNLayer(nn.Module):
    max_ell: int
    output_irreps: e3nn.Irreps  # 例如 64 * e3nn.Irreps("0e + 1o + 2e")
    num_mlp_layers: int
    dim_mlp: int
    radial_basis: Callable[[jax.Array], jax.Array]
    n_radial_basis: int
    mlp_act_fn: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    even_activation: Callable[[jax.Array], jax.Array] = jax.nn.silu
    odd_activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    gate_activation: Callable[[jax.Array], jax.Array] = jax.nn.silu

    @nn.compact
    def __call__(
            self,
            x: e3nn.IrrepsArray,
            vectors: e3nn.IrrepsArray,  # shape [n_edges, 3]
            r: jax.Array,
            senders: Sequence[int],
            receivers: Sequence[int],
            ) -> e3nn.IrrepsArray:
        num_nodes = x.shape[0]
        output_irreps = e3nn.Irreps(self.output_irreps).regroup()

        # Input Linear
        linear_up = e3nn_flax.Linear(self.output_irreps, name="linear_up")
        messages = linear_up(x)[senders]  # 直接调用，输入 irreps 自动适配
        
        # Angular part
        direct_part = messages.filter(output_irreps + "0e")
        # 球谐函数（l >= 1）
        Y = e3nn.spherical_harmonics(
            [l for l in range(1, self.max_ell + 1)],
            vectors,
            normalize=True,
            normalization="component",
            )
        # 张量积生成新的角度依赖特征
        tp_part = e3nn.tensor_product(
            messages, Y, filter_ir_out=output_irreps + "0e")
        messages = e3nn.concatenate([direct_part, tp_part]).regroup()

        # ---------- 径向部分（此处省略，可自行添加） ----------
        # 原代码中径向部分缺失，可按需补充（例如通过 MLP 生成权重并与 messages 相乘）
        mix = e3nn_flax.MultiLayerPerceptron(
            self.num_mlp_layers * (self.dim_mlp,) + (messages.irreps.num_irreps,),
            self.mlp_act_fn,
            output_activation=False
            )(self.radial_basis(r[:, 0], self.n_radial_basis))
        
        # Product of radial and angular part
        messages = messages * mix

        # ---------- 准备跳跃连接的目标 irreps ----------
        irreps = output_irreps.filter(keep=messages.irreps)
        num_nonscalar = irreps.filter(drop="0e + 0o").num_irreps
        irreps = irreps + e3nn.Irreps(f"{num_nonscalar}x0e").simplify()

        # ---------- 跳跃连接 ----------
        linear_skip = e3nn_flax.Linear(irreps, name="linear_skip")
        skip = linear_skip(x)

        # ---------- 消息聚合 ----------
        x = e3nn.scatter_sum(messages, dst=receivers, output_size=num_nodes)
        x = x / jnp.sqrt(num_nodes)  # 注意：此处应使用平均邻居数，这里仅示例

        # ---------- 下投影 ----------
        linear_down = e3nn_flax.Linear(irreps, name="linear_down")
        x = linear_down(x)

        # ---------- 残差连接 ----------
        x = x + skip

        # ---------- 等变门控激活 ----------
        x = e3nn.gate(
            x,
            even_act=self.even_activation,
            odd_act=self.odd_activation,
            even_gate_act=self.gate_activation,
            )

        return x