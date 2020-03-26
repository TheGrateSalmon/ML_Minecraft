from pathlib import Path

import numpy as np


def find_dir(path: Path, desired_dir="data"):
    # find data directory by searching hierarchy of directories
    # should find "data" directory
    max_levels = 3

    i = 0
    while i < max_levels:
        # "data" directory is found
        if path / desired_dir in path.iterdir():
            path = path / desired_dir
            i = max_levels + 1   # condition found, break out of while
        # move up a directory
        else:
            i += 1
            path = path.parent    # try again, but one directory up

        # don't check too far up hierarchy
        if i == max_levels:
            raise FileNotFoundError

    return path


class Config:
    """Contains variables such as data directories or hyperparameters."""

    def __init__(self, mode: str="train"):
        self.mode = mode
        #self.data_dir = Path("E:\\User\\Documents\\Projects\\data\\minecraft\\")
        self.data_dir = find_dir(Path().cwd())/"minecraft"

        # number of blocks in a chunk for each axis (x,y, and z)
        self.num_x_in_chunk = 16
        self.num_z_in_chunk = 16
        self.num_y_in_chunk = 256
        self.num_features = np.load(self.all_raw_files[0]).shape[1]

        # model parameters
        self.z_dim = 100

        # hyperparameters
        self.batch_size = 64
        self.epochs = 100
        self.lr = 0.001
        self.loss = "chamfer"   # either 'chamfer' or 'emd'
        self.normal_mu = 0.0
        self.normal_std = 0.2

        # optimizer hyparameters
        self.eg_optimizer_type = "Adam"
        self.eg_optimizer_hparams = {"lr": 0.0005, 
                                     "weight_decay": 0.0,
                                     "betas": [0.9, 0.999],
                                     "amsgrad": False}

        self.d_optimizer_type = "Adam"
        self.d_optimizer_hparams = {"lr": 0.0005, 
                                    "weight_decay": 0.0,
                                    "betas": [0.9, 0.999],
                                    "amsgrad": False}

    @property
    def all_raw_files(self):
        """Recursively finds all NumPy files in the raw/ folder of data directory.

        https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob
        """
        
        files = list((self.data_dir/"raw").rglob("*.npy"))
        
        return files

    @property
    def all_region_files(self):
        """Recursively finds all MCA files in the raw/ folder of data directory.

        https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob
        """
        
        files = []
        for extension in ["mca", "mcr"]:
            files.extend((self.data_dir/"regions").rglob(f"*.{extension}"))
        
        return files

    @property
    def total_points(self):
        return self.num_x_in_chunk * self.num_z_in_chunk * self.num_y_in_chunk

    @property
    def device(self):
        """Check if CUDA device is available. If not, default to CPU."""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_hparams(self):
        print(f"""Encoder, Generator hyperparameters
                  ==================================
                  optimizer: {self.eg_optimizer_type}
                  learning rate: {self.eg_optimizer_hparams['lr']}
                  weight decay: {self.eg_optimizer_hparams['weight_decay']}

                  Discriminator hyperparameters
                  =============================
                  optimizer: {self.d_optimizer_type}
                  learning rate: {self.eg_optimizer_hparams['lr']}
                  weight decay: {self.eg_optimizer_hparams['weight_decay']}

                  Training hyperparameters
                  ========================
                  batch size: {self.batch_size}
                  epochs: {self.epochs}
               """)