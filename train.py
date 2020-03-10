from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.configs import Config
from datasets import MCDataset
from models import GAN


if __name__ == "__main__":
    # load hyperparameters and paths
    cfg = Config(mode="train")

    # load the training/validation datasets
    mc_dataset = MCDataset(cfg)
    mc_dataloader = DataLoader(mc_dataset, batch_size=cfg.batch_size)

    # initialize the network for training


    # train
    for epoch in cfg.epochs:
        for _ in mc_dataloader:
            # random noise from latent space
            noise = torch.randn(cfg.batch_size, 1, cfg.latent_dim)
