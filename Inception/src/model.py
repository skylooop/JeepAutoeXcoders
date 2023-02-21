import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax

import typing as tp


class TrainState(train_state.TrainState):
    batch_stats: tp.Any

