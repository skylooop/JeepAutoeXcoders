import numpy as np
import torch
import typing as tp
import torchvision
from torch.utils.data import DataLoader
from collections import namedtuple
from torchvision.datasets import CIFAR10


def numpy_collate(batch: torch.Tensor) -> tp.Any:
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_datasets(root: str, train_transform: torchvision.transforms.Compose,
                            test_transform: tp.Callable[[np.ndarray], np.ndarray]) -> tp.Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_dataset = CIFAR10(root=root, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=root, train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    test_set = CIFAR10(root=root, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set,
                               batch_size=128,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate,
                               num_workers=8,
                               persistent_workers=True)
    val_loader   = DataLoader(val_set,
                                batch_size=128,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=numpy_collate,
                                num_workers=8,
                                persistent_workers=True)
    test_loader  = DataLoader(test_set,
                                batch_size=128,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=numpy_collate,
                                num_workers=8,
                                persistent_workers=True)
    
    return train_loader, val_loader, test_loader