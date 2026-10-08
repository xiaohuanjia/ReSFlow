import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from ._base import register_dataset
from .prepare import prepare_text8


@register_dataset('text8')
class Text8Dataset(Dataset):
    def __init__(self, root, split, vocab_size=27, seq_len=256):
        """
        seq_len should include context length. Example: seq_len=512 for modeling 256 chars with 256 char of context.
        context is only used for correct preparation of val/test sets.
        """
        self.root = root
        self.split = split
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        fname = os.path.join(root, f'{split}.bin')
        prepare_text8(root)
        self.data = np.memmap(fname, np.uint16, 'r')

    def __getitem__(self, index):
        seq = torch.from_numpy(self.data[index: index + self.seq_len].astype(np.int64))
        seq = F.one_hot(seq, self.vocab_size).float()
        return (seq,)

    def __len__(self):
        return self.data.size - self.seq_len
