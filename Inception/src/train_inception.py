from absl import flags, app
import os
from PIL import Image
from collections import defaultdict, namedtuple
import numpy as np


# JAX
import jax
import jax.numpy as jnp

# Neural Networks for JAX
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

# Optimizers for JAX
import optax

# Torch utils
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", default="/home/bobrin_m_s/Projects/tmp", help="Save path for CIFAR10.")


dataset_info = namedtuple("Info", ['mean', 'std'])

def main(_):
    main_rng = jax.random.PRNGKey(42)
    train_dataset = CIFAR10(root=FLAGS.dataset_path, train=True, download=True)
    info = dataset_info((train_dataset.data / 255.0).mean(axis=(0, 1, 2)), 
                        (train_dataset.data / 255.0).std(axis=(0, 1, 2)))
    print(train_dataset)



if __name__ == "__main__":
    app.run(main)