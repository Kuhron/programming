import random

import Minecraft.mcInterface as mcInterface


SAVE_DIR = "C:/Users/Wesley/AppData/Roaming/.minecraft/saves/"
LOADNAME = SAVE_DIR + "PythonExperimentation"


def place_blocks(mclevel):
    block_info = {"B": 57, "D": 0}
    for i in range(5000):
        x = random.randint(-100, 100)
        y = random.randint(60, 100)
        z = random.randint(-100, 100)
        print("setting ({}, {}, {}) to {}".format(x, y, z, block_info))
        mclevel.set_block(x, y, z, block_info)


def place_pillar(mclevel):
    block_info = {"B": 57, "D": 0}
    for i in range(60, 80):
        x = 0
        y = i
        z = 0
        print("setting ({}, {}, {}) to {}".format(x, y, z, block_info))
        mclevel.set_block(x, y, z, block_info)


def main():
    try:
        level = mcInterface.SaveFile(LOADNAME)
        print("!!!\n" * 4 + "finished building level!\n" + "!!!\n" * 4)
    except IOError:
        print('File name invalid or save file otherwise corrupted. Aborting')
        return None
    # place_blocks(level)
    # place_pillar(level)
    print(level.block(0, 70, 0))

    # success = level.write()
    # if success:
    #     print("saved successfully")
    # else:
    #     print("could not save")


if __name__ == '__main__':
    main()
