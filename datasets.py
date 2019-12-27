from pathlib import Path

import numpy as np
import torch
from torch.utils import Dataset, DataLoader

from utils.configs import Config

class MCDataset:
    def __init__(self, config: Config):
        # should chunks be saved in compressed npz or as separate files?
        self.chunks = [np.load(f) for f in (config.data_dir/"filled"/config.mode).iterdir() if f.is_file()]
        self.labels = [np.load(f) for f in (config.data_dir/"labels"/config.mode).iterdir() if f.is_file()]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # label should be last column of array
        return self.chunks[idx], self.labels[idx]
