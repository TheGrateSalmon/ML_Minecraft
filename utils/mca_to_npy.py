from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm
sys.path.append(Path("E:\\User\\Documents\\Projects\\minecraft\\jnbt").as_posix())
import jnbt
from jnbt.mc.world import Region    # https://github.com/theJ8910/jnbt/blob/edfe4f1964d28be082606b428d8fdd4009d431c8/jnbt/mc/world/base.py

from configs import Config


def get_features(file_name: Path, features: list):
    """Loads a Minecraft world's chunks and keeps specified information."""
    pass


if __name__ == "__main__":

    cfg = Config()

    count = 1
    for region_file in tqdm(cfg.all_region_files, total=len(cfg.all_region_files), desc="region"):
        region = Region(region_file)

        for chunk in region.iterChunks():
            #blocks = [block for block in chunk.iterBlocks()]
            blocks = np.vstack([[block.x, block.z, block.y, block.id] for block in chunk.iterBlocks()])
            np.save(cfg.data_dir/"raw"/f"raw_chunk_{count}.npy", blocks)
            count += 1
