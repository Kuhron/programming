import random
import time

from mcpi.minecraft import Minecraft
import mcpi.block as block
from mcutil import set_blocks_cuboid, get_spiral_xzs


def get_spiral_xzs(apothem):
    # spiral from center
    # the run lengths are 1,1,2,2,3,3,4,4,... in the four directions
    directions = [(1,0),(0,1),(-1,0),(0,-1)]
    start_xz = (0,0)
    xz = start_xz
    yield start_xz

    step_i = 0
    while True:
        dx = abs(xz[0] - start_xz[0])
        if dx >= apothem:
            break

        direction_i = step_i % 4
        direction = directions[direction_i]
        run_length = step_i // 2 + 1  # 0>1, 1>1, 2>2, 3>2, 4>3, ...
        for run_i in range(run_length):
            x,z = xz
            dir_x, dir_z = direction
            x += dir_x
            z += dir_z
            xz = (x,z)
            yield xz
        step_i += 1


def generate_maze(mc, player_pos, block_type):
    px, py, pz = player_pos
    px = int(px)
    py = int(py)
    pz = int(pz)
    mc.postToChat("player_pos is {}".format(player_pos))

    cell_size = 4  # size 4 means 3x3 space with wall of thickness 1, tessellate the plane with these
    height = 5
    apothem = 10
    for cell_xz in get_spiral_xzs(apothem):
        time.sleep(0.01)  # allow the game to load it more easily if i's not all at once?
        mc.postToChat("cell_xz = {}".format(cell_xz))
        cx, cz = cell_xz
        bottom_left_coords = (px+cx*cell_size, py, pz+cz*cell_size)
        # make the left and bottom walls only, the top and right really belong to other cells
        # for ease, just say left = x and bottom = z
        blx, bly, blz = bottom_left_coords
        left_wall_xyz0 = bottom_left_coords
        left_wall_xyz1 = (blx + cell_size - 1, bly + height - 1, blz)
        left_wall_interior_xyz0 = (left_wall_xyz0[0]+1, left_wall_xyz0[1], left_wall_xyz0[2])
        left_wall_interior_xyz1 = (left_wall_xyz1[0]-1, left_wall_xyz1[1], left_wall_xyz1[2])
        bottom_wall_xyz0 = bottom_left_coords
        bottom_wall_xyz1 = (blx, bly + height - 1, blz + cell_size - 1)
        bottom_wall_interior_xyz0 = (bottom_wall_xyz0[0], bottom_wall_xyz0[1], bottom_wall_xyz0[2]+1)
        bottom_wall_interior_xyz1 = (bottom_wall_xyz1[0], bottom_wall_xyz1[1], bottom_wall_xyz1[2]-1)
        ceiling_xyz0 = (blx, bly + height, blz)
        ceiling_xyz1 = (blx + cell_size - 1, bly + height, blz + cell_size - 1)

        # randomly choose whether to fill in each wall or leave it open
        left_wall_open = random.random() < 0.5
        bottom_wall_open = random.random() < 0.5
        # fill in both walls by default, then refill the interior cuboid with air if it's open

        set_blocks_cuboid(mc, left_wall_xyz0, left_wall_xyz1, block_type)
        # lwx0, lwy0, lwz0 = left_wall_xyz0
        # lwx1, lwy1, lwz1 = left_wall_xyz1
        # mc.setBlocks(lwx0, lwy0, lwz0, lwx1, lwy1, lwz1, block_type)
        if left_wall_open:
            # lix0, liy0, liz0 = left_wall_interior_xyz0
            # lix1, liy1, liz1 = left_wall_interior_xyz1
            # mc.setBlocks(lix0, liy0, liz0, lix1, liy1, liz1, block_type)
            set_blocks_cuboid(mc, left_wall_interior_xyz0, left_wall_interior_xyz1, block.AIR)

        set_blocks_cuboid(mc, bottom_wall_xyz0, bottom_wall_xyz1, block_type)  # doesn't work
        # lwx0, lwy0, lwz0 = bottom_wall_xyz0
        # lwx1, lwy1, lwz1 = bottom_wall_xyz1
        # mc.setBlocks(lwx0, lwy0, lwz0, lwx1, lwy1, lwz1, block_type)
        if bottom_wall_open:
            # lix0, liy0, liz0 = bottom_wall_interior_xyz0
            # lix1, liy1, liz1 = bottom_wall_interior_xyz1
            # mc.setBlocks(lix0, liy0, liz0, lix1, liy1, liz1, block_type)
            set_blocks_cuboid(mc, bottom_wall_interior_xyz0, bottom_wall_interior_xyz1, block.AIR)

        set_blocks_cuboid(mc, ceiling_xyz0, ceiling_xyz1, block_type)



if __name__ == "__main__":
    mc = Minecraft()

    player_pos = mc.player.getPos()
    # block_type = block.WOOL_YELLOW
    # block_type = block.WOOD_PLANKS_JUNGLE
    block_type = block.GLASS
    generate_maze(mc, player_pos, block_type)
