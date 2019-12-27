from pathlib import Path

import numpy as np
from tqdm import tqdm

from configs import Config


def train_test_split(files: list):
    train_percent = 0.75    # test_percent = 1-train_percent
    split_idx = int(len(files) * train_percent)

    return files[:split_idx], files[split_idx:]


# a point has (x, y, z, block_id, biome)?
def complete_chunk(chunk: np.array, full_chunk: np.array, config: Config):
    """Fills in missing blocks of a chunk so that it has 16x16x256 blocks."""

    # chunk is incomplete, pad matrix with air blocks at all x,y,z
    if chunk.shape[0] != full_chunk.shape[0]:
        # find boundaries and get coordinate values
        x_min, x_max = chunk[:, 0].min(), chunk[:, 0].max()
        z_min, z_max = chunk[:, 1].min(), chunk[:, 1].max()

        idx = 0
        for x in range(x_min, x_max+1):
            for z in range(z_min, z_max+1):
                for y in range(cfg.num_y_in_chunk):
                    # check if point (x,y,z) is already a point in the chunk
                    # https://stackoverflow.com/questions/51031140/check-whether-2d-array-contains-a-specific-1d-array-in-numpy
                    entry_check = np.isin(chunk[:, :3], np.array([x,y,z]), assume_unique=True)
                    row_check = np.all(entry_check, axis=1)

                    if np.any(row_check):
                        full_chunk[idx] = chunk[np.where(row_check)][0]
                    else:
                        full_chunk[idx] = np.hstack([np.array([x,z,y]), np.zeros((cfg.num_features-3))])
                        # print(full_chunk[idx])
                    idx += 1

    # else return the original chunk
    else:
        full_chunk = chunk

    return full_chunk


def save_chunk(chunk: np.array, split_name: str, config: Config):
    """Saves a chunk to train/test directory."""

    save_location = cfg.data_dir / "filled" / split_name
    np.save(save_location, chunk)


if __name__ == "__main__":

    cfg = Config()

    train_files, test_files = train_test_split(cfg.all_raw_files)

    # ensure all chunks are same size and save the new chunks
    empty_chunk = np.zeros((cfg.num_x_in_chunk*cfg.num_z_in_chunk*cfg.num_y_in_chunk, cfg.num_features))
    count = 1
    for subset_name, subset in {"train": train_files, "test": test_files}.items():
        for raw_chunk_file in tqdm(subset, desc=subset_name):
            raw_chunk = np.load(raw_chunk_file)
            filled_chunk = complete_chunk(raw_chunk, empty_chunk, cfg)

            np.save(cfg.data_dir/"filled"/subset_name/f"chunk_{count}.npy", filled_chunk)
            count += 1
