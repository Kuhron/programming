import random
import time

from mcpi.minecraft import Minecraft
import mcpi.block as block
from mcutil import set_blocks_cuboid, get_spiral_xzs


def generate_forest(mc, player_pos, block_type):
    px, py, pz = player_pos
    px = int(px)
    py = int(py)
    pz = int(pz)

    spacing = 8
    height = 128
    apothem = 100
    for xz in get_spiral_xzs(apothem):
        time.sleep(0.01)
        mc.postToChat("cell {}".format(xz))
        x, z = xz
        x *= spacing
        z *= spacing
        x += px
        y = py
        z += pz
        top_y = min(255, y + height)
        xyz0 = (x, y, z)
        xyz1 = (x, top_y, z)
        set_blocks_cuboid(mc, xyz0, xyz1, block_type)


if __name__ == "__main__":
    mc = Minecraft()

    player_pos = mc.player.getPos()
    block_type = block.WOOD
    generate_forest(mc, player_pos, block_type)
