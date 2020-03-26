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
        self.latent_dim = 100

        # hyperparameters
        if self.mode == "train":
            self.batch_size = 64
            self.epochs = 100
            self.lr = 0.001


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
