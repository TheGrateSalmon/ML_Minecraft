"""Loads a Minecraft Region into a NumPy array."""

import numpy as np
from tqdm import tqdm
import pptk
# documentation for jnbt can be found here:     https://github.com/theJ8910/jnbt
import jnbt     # for reading Minecraft's .mca files


#####################
#                   #
#     LOAD_DATA     #
#                   #
#####################
def load_data(world_name: str):
    r"""Reads in data from a Minecraft world chunk region."""
    
    generate = False
    do_filter = False

    if generate:
        world = jnbt.getWorld(world_name)

        blocks = np.empty((200_000_000,4))
        num_blocks = 0

        for block in tqdm(world[ jnbt.DIM_OVERWORLD ].iterBlocks()):
                blocks[num_blocks, :] = np.array([block.x, block.z, block.y, block.id])
                num_blocks += 1
        blocks = blocks[:num_blocks, :]
        np.save(world_name + str(jnbt.DIM_NETHER) + '_blocks.npy', blocks)

    blocks = np.load(world_name + str(jnbt.DIM_NETHER) + '_blocks.npy')

    num_blocks = blocks.shape[0]
    total_blocks = num_blocks

    print(f'total blocks: {total_blocks}')
    print('='*20)
    print(f'blocks: {num_blocks}\t({round(num_blocks / total_blocks, 4)*100}%)\n')

    blocks = blocks[blocks[:, 3] != 0]      # air
    blocks = blocks[blocks[:, 3] != 8]      # flowing water
    blocks = blocks[blocks[:, 3] != 9]      # water
    blocks = blocks[blocks[:, 3] != 10]      # flowing lava
    blocks = blocks[blocks[:, 3] != 11]      # lava
    if do_filter:
        blocks = blocks[blocks[:, 3] != 1]      # stone
        blocks = blocks[blocks[:, 3] != 14]      # gold ore
        blocks = blocks[blocks[:, 3] != 15]      # iron ore
        blocks = blocks[blocks[:, 3] != 16]      # coal ore
        blocks = blocks[blocks[:, 3] != 21]      # lapis lazuli ore
        blocks = blocks[blocks[:, 3] != 56]      # diamond ore
        blocks = blocks[blocks[:, 3] != 73]      # redstone ore
        blocks = blocks[blocks[:, 3] != 74]      # glowing redstone ore
        blocks = blocks[blocks[:, 3] != 129]      # emerald ore
    blocks = blocks[blocks[:, 2] == 48]
    print(f'Size of subset of blocks: {blocks.shape[0]}\t({round(blocks.shape[0] / total_blocks, 4)*100}%)\n')

    # print(f'Constructing air point cloud . . . ')
    # air = air[air[:, 2] < 63]
    # v1 = pptk.viewer(air[:, :3])
    # v1.attributes(air[:, 3], air[:, 2])
    # v1.set(point_size=.2)
    # v1.wait()
    # print(f'Finished constructing air point cloud.\n')

    print(f'Constructing blocks point cloud . . . ')
    v2 = pptk.viewer(blocks[:, :3])

    # block ID, z
    v2.attributes(blocks[:, 3], blocks[:, 2])
    v2.set(point_size=.2)
    print(f'Finished constructing blocks point cloud.\n')

if __name__ == '__main__':
    load_data('New world')
