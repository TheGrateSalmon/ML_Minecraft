from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.configs import Config
from datasets import MCDataset
import models


if __name__ == "__main__":

    cfg = Config(mode="train")
