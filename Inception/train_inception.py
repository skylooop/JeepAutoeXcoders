from absl import flags, app
import os
from PIL import Image
from collections import defaultdict, namedtuple
import numpy as np

# JAX
import jax
import jax.numpy as jnp

#Flax
import flax.linen as nn

# Torch utils
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import warnings

from utils import data
import matplotlib.pyplot as plt
import typing as tp

from src.model import TrainerModule, GoogleNet


warnings.filterwarnings("ignore")
FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", default="/home/bobrin_m_s/Projects/tmp", help="Save path for CIFAR10.")
flags.DEFINE_bool("visualize", default=True, help="Whether to save visualization samples.")
flags.DEFINE_integer("seed", default=42, help="Seed for initialization.")

dataset_info = namedtuple("Info", ['mean', 'std'])

class PreprocessingCIFAR:
    def __init__(self, info: namedtuple) -> None:
        self.info = info

    def to_numpy(self, image: np.ndarray) -> np.ndarray:
        img = np.array(image, dtype=np.float32)
        img = (img / 255.0 - self.info.mean) / self.info.std
        return img
    
    @staticmethod
    def plotting(train_dataset: tp.Any, test_transform: tp.Callable[[np.ndarray], np.ndarray]) -> None:
        NUM_IMAGES = 4
        images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
        orig_images = [Image.fromarray(train_dataset.data[idx]) for idx in range(NUM_IMAGES)]
        orig_images = [test_transform(img) for img in orig_images]

        imgs = np.stack(images + orig_images, axis=0)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        img_grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, pad_value=0.5)
        img_grid = img_grid.permute(1, 2, 0)

        plt.figure(figsize=(8,8))
        plt.title("Augmented examples on CIFAR10 (Top: augmented, bottom: original)")
        plt.imshow(img_grid)
        plt.savefig("assets/sample_images.jpg")

def main(_):
    main_rng = jax.random.PRNGKey(FLAGS.seed)
    train_dataset = CIFAR10(root=FLAGS.dataset_path, train=True, download=True)
    info = dataset_info((train_dataset.data / 255.0).mean(axis=(0, 1, 2)), 
                        (train_dataset.data / 255.0).std(axis=(0, 1, 2)))
    
    preprocess = PreprocessingCIFAR(info=info)
    test_transform = preprocess.to_numpy

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        preprocess.to_numpy
    ])
    train_loader, val_loader, test_loader = data.get_datasets(root=FLAGS.dataset_path, train_transform=train_transform, test_transform=test_transform)

    imgs, _ = next(iter(train_loader))
    print(f"Batch mean of dataset: {imgs.mean(axis=(0,1,2))}")
    print(f"Batch std  of dataset: {imgs.std(axis=(0,1,2))}")

    if FLAGS.visualize:
        preprocess.plotting(train_dataset, test_transform)

    def train_classifier(*args, num_epochs=200, **kwargs):
        trainer = TrainerModule(*args, **kwargs)
        if not trainer.checkpoint_exists():
            trainer.train_model(train_loader, val_loader, num_epochs=num_epochs)
            trainer.load_model()
        else:
            trainer.load_model(pretrained=True)

        val_acc = trainer.eval_model(val_loader)
        test_acc = trainer.eval_model(test_loader)
        return trainer, {'val': val_acc, 'test': test_acc}

    googlenet_trainer, googlenet_results = train_classifier(model_class=GoogleNet,
                                                        model_name="GoogleNet",
                                                        model_hparams={"num_classes": 10,
                                                                       "act_fn": nn.relu},
                                                        optimizer_name="adamw",
                                                        optimizer_hparams={"lr": 1e-3,
                                                                           "weight_decay": 1e-4},
                                                        exmp_imgs=jax.device_put(
                                                            next(iter(train_loader))[0]),
                                                        num_epochs=200)
    print("GoogleNet Results", googlenet_results)

if __name__ == "__main__":
    app.run(main)