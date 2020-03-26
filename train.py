from pathlib import Path
import itertools as it

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils import Config
from utils import ChamferLoss as CL, EarthMoversDistance as EMD
from datasets import MCDataset
from models import Generator, Encoder, Discriminator


if __name__ == "__main__":
    # load hyperparameters and paths
    config = Config(mode="train")

    # load the training/validation datasets
    mc_dataset = MCDataset(config)
    mc_dataloader = DataLoader(mc_dataset, batch_size=config.batch_size)

    # initialize the networks for training and send to GPU if available
    generator = Generator(z_dim=config.z_dim, out_features=config.total_points).to(config.device)
    encoder = Encoder(z_dim=config.z_dim).to(config.device)
    discriminator = Discriminator(z_dim=config.z_dim).to(config.device)

    # initialize loss and send to GPU if available
    if config.loss not in {"chamfer", "emd"}:
        raise ValueError(f"{config.loss} is not a valid loss function. Choose 'chamfer' or 'emd'")
    reconstruction_loss = CL() if config.loss == "chamfer" else EMD()
    reconstruction_loss.to(config.device)

    # initialize noise tensors and send to GPU if available
    fixed_noise = torch.FloatTensor(config.batch_size, config.z_dim, 1)
    fixed_noise.normal_(mean=config.normal_mu, std=config.normal_std)
    noise = torch.FloatTensor(config.batch_size, config.z_dim)

    fixed_noise = fixed_noise.to(device)
    noise = noise.to(device)

    # initialize optimizers for each network
    eg_optim = getattr(optim, config.e_optimizer_type)
    eg_optim = eg_optim(it.chain(encoder.parameters(), generator.parameters()),
                        **config.eg_optimizer_hparams)

    d_optim = getattr(optim, config.d_optimizer_type)
    d_optim = d_optim(discriminator.parameters(),
                      **config.d_optimizer_hparams)

    # train the networks
    config.print_hparams()
    for epoch in range(cfg.epochs):
        for _ in mc_dataloader:
            # random noise from latent space
            noise = torch.randn(config.batch_size, 1, config.z_dim)
