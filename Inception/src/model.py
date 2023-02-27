import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax

import typing as tp


class TrainState(train_state.TrainState):
    batch_stats: tp.Any

class TrainerModule:
    def __init__(self, model_name: str, model_class: nn.Module, 
                 model_hparams:  dict, optimizer_name: str, 
                 optimizer_hparams: dict, expmp_imgs: tp.Any, seed: int = 42) -> None:
        