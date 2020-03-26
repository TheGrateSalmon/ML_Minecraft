from pathlib import Path
import itertools as it
from datetime import datetime

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
        start_epoch_time = datetime.now()

        # set networks for training
        generator.train()
        encoder.train()
        discriminator.train()

        # initialize loss values
        total_loss_eg, total_loss_d = 0.0, 0.0

        for i, points_data in enumerate(mc_dataloader, 1):
            chunk, _ = points_data
            chunk = chunk.to(config.device)

            # change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if chunk.size(-1) == 3:
                chunk.transpose_(chunk.dim()-2, chunk.dim()-1)

            codes, _, _ = encoder(chunk)
            noise.normal_(mean=config.normal_mu, std=config.normal_std)
            synth_logit = discriminator(codes)
            real_logit = discriminator(noise)
            loss_d = torch.mean(synth_logit) - torch.mean(real_logit)

            alpha = torch.rand(config.batch_size, 1).to(config.device)
            differences = codes - noise
            interpolates = noise + alpha * differences
            disc_interpolates = discriminator(interpolates)

            gradients = grad(outputs=disc_interpolates,
                             inputs=interpolates,
                             grad_outputs=torch.ones_like(disc_interpolates).to(config.device),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]
            slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
            gradient_penalty = ((slopes - 1) ** 2).mean()
            loss_gp = config.gp_lambda * gradient_penalty

            loss_d += loss_gp

            # reset disciminator gradients from previous epochs
            d_optim.zero_grad()
            discriminator.zero_grad()

            loss_d.backward(retain_graph=True)
            total_loss_d += loss_d.item()
            d_optim.step()

            # encoder, generator part of training
            chunk_recon = generator(codes)

            loss_e = torch.mean(config.reconstruction_coeff * reconstruction_loss(chunk.permute(0, 2, 1) + 0.5, chunk_recon.permute(0, 2, 1) + 0.5))

            synth_logit = discriminator(codes)

            loss_g = -torch.mean(synth_logit)

            loss_eg = loss_e + loss_g
            # reset encoder, generator gradients from last epoch
            eg_optim.zero_grad()
            encoder.zero_grad()
            generator.zero_grad()

            loss_eg.backward()
            total_loss_eg += loss_eg.item()
            eg_optim.step()

            print(f"[{epoch}: ({i})] "
                  f"Loss_D: {loss_d.item():.4f} "
                  f"(GP: {loss_gp.item(): .4f}) "
                  f"Loss_EG: {loss_eg.item():.4f} "
                  f"(REC: {loss_e.item(): .4f}) "
                  f"Time: {datetime.now() - start_epoch_time}")

        # show results
        generator.eval()
        encoder.eval()
        discriminator.eval()

        with torch.no_grad():
            fake = generator(fixed_noise).data.cpu().numpy()
            codes, _, _ = encoder(X)
            chunk_recon = generator(codes).data.cpu().numpy()

            # TODO: PPTK stuff

        if epoch % config.save_frequency == 0:
            # save models
            torch.save(G.state_dict(), config.weights_path/f"{epoch:05}_generator.pth")
            torch.save(D.state_dict(), config.weights_path/f"{epoch:05}_discriminator.pth")
            torch.save(E.state_dict(), config.weights_path/f"{epoch:05}_encoder.pth")

            # save optimizers
            torch.save(eg_optim.state_dict(), config.weights_path/f"{epoch:05}_ego.pth")
            torch.save(d_optim.state_dict(), config.weights_path/f"{epoch:05}_do.pth")
