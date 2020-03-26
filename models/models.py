"""Implementation from https://github.com/MaciejZamorski/3d-AAE/blob/master/models/aae.py"""

import torch
import torch.nn as nn


class Encoder(nn.Module):  
    def __init__(self, z_dim: int=100, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.in_features = in_features
        self.z_dim = z_dim

        self.use_bias = kwargs.get("use_bias", True)

        self.conv = nn.Sequential(
                                  nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                                            bias=self.use_bias),
                                  nn.ReLU(inplace=True),

                                  nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                                            bias=self.use_bias),
                                  nn.ReLU(inplace=True),

                                  nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                                            bias=self.use_bias),
                                  nn.ReLU(inplace=True),

                                  nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                                            bias=self.use_bias),
                                  nn.ReLU(inplace=True),

                                  nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                                            bias=self.use_bias),
                                  )

        self.fc = nn.Sequential(
                                nn.Linear(512, 256, bias=True),
                                nn.ReLU(inplace=True)
                                )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class Generator(nn.Module):
    """

    Note: out_features is the number of output POINTS of the generator.
    """

    def __init__(self, z_dim: int=100, out_features: int=16*16*256, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.z_dim = z_dim
        self.out_features = out_features

        self.use_bias = kwargs.get("use_bias", True)

        self.model = nn.Sequential(
                                   nn.Linear(in_features=self.z_dim, out_features=64, bias=self.use_bias),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(in_features=1024, out_features=out_features * 3, bias=self.use_bias)
                                   )

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, 3, self.out_features)
        
        return output


class Discriminator(nn.Module):
    def __init__(self, z_dim: int=100, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        # features_in=z_dim since the discrinator takes input from latent space
        self.z_dim = z_dim

        self.model = nn.Sequential(
                                   nn.Linear(self.z_dim, 512, bias=True),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(512, 512, bias=True),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(512, 128, bias=True),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(128, 64, bias=True),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(64, 1, bias=True)
                                   )

    def forward(self, x):
        logit = self.model(x)
        
        return logit
