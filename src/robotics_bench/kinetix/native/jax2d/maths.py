import jax
import jax.numpy as jnp


def rmat(angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])


def sv_cross(x, v):
    return jnp.array([-x * v[1], x * v[0]])


def vs_cross(v, x):
    return jnp.array([x * v[1], -x * v[0]])


def zero_to_one(x):
    # We use this to avoid NaNs cropping up in masked out shapes
    return jax.lax.select(x == 0, jnp.ones_like(x), x)
