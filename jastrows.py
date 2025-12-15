import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional
from flax.linen.initializers import constant

def simple_ee_cusp_fun(
    r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return - (cusp * alpha**2) / (alpha + r)


class Jastrow_ee(nn.Module):
    jastrow_fn: Callable[[jnp.ndarray, float, jnp.ndarray], jax.Array] = simple_ee_cusp_fun
    
    def setup(self):
        # 使用 Flax 的参数定义方式
        self.ee_par = self.param(
            'ee_par', constant(1.0), (1,)
            )
        self.ee_anti = self.param(
            'ee_anti', constant(1.0), (1,)
            )
    
    def __call__(self, nspins: jax.Array, r_ee: jax.Array) -> jax.Array:
        """Jastrow factor for electron-electron cusps."""
        # 分割电子-电子距离矩阵
        r_ees = [
            jnp.split(r, [nspins[0]], axis=1)
            for r in jnp.split(r_ee, [nspins[0]], axis=0)
        ]
        
        # 处理同自旋电子对
        r_ees_parallel = jnp.concatenate([
            r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
            r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
        ])
        
        # 计算同自旋Jastrow因子
        if r_ees_parallel.shape[0] > 0:
            jastrow_ee_par = jnp.sum(
                self.jastrow_fn(r_ees_parallel, 0.25, self.ee_par)
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)
        
        # 处理异自旋电子对
        if r_ees[0][1].shape[0] > 0:
            jastrow_ee_anti = jnp.sum(self.jastrow_fn(r_ees[0][1], 0.5, self.ee_anti))
        else:
            jastrow_ee_anti = jnp.asarray(0.0)
        
        return jastrow_ee_anti + jastrow_ee_par
    