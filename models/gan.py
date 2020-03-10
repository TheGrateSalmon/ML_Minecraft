import torch
import torch.nn as nn

class GAN(nn.Module):
    def __init__(self, **kwargs):
        super(GAN, self).__init__()
