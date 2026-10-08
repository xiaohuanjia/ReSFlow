
import numpy as np
import torch
from torch.utils.data import Dataset

from ._base import register_dataset
from .prepare import BMNIST_URL, prepare_bmnist


@register_dataset('bmnist')
class BinaryMNIST(Dataset):
    """
    Binarized MNIST dataset.
    """
    data_url = BMNIST_URL

    def __init__(self, root, split, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        cache = prepare_bmnist(root, split)
        self.data = torch.from_numpy(np.load(cache, allow_pickle=False))

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.stack([x, 1 - x], dim=-1)
        if self.transform is not None:
            x = self.transform(x)
        return (x,)

    def __len__(self):
        return self.data.size(0)
