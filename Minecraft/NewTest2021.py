import os
import random


# documentation of the file format:
# https://minecraft.gamepedia.com/Region_file_format
# https://minecraft.gamepedia.com/Anvil_file_format
# the .mca are Anvil format, which has a different coordinate order but otherwise seems to be the same as Region format

# X is east, Y is up, Z is south


def get_region_xz(point_xz):
    chunk_x, chunk_z = point_xz
    region_x = chunk_x // 32
    region_z = chunk_z // 32
    return region_x, region_z


def to_bit_array(bytestring):
    res = []
    for byte in bytestring:
        assert type(byte) is int, byte
        assert 0 <= byte < 256, byte
        rest = byte
        a, rest = divmod(rest, 2 ** 7)
        b, rest = divmod(rest, 2 ** 6)
        c, rest = divmod(rest, 2 ** 5)
        d, rest = divmod(rest, 2 ** 4)
        e, rest = divmod(rest, 2 ** 3)
        f, rest = divmod(rest, 2 ** 2)
        g, rest = divmod(rest, 2 ** 1)
        h = rest
        arr8 = [a,b,c,d,e,f,g,h]
        assert all(x in [0,1] for x in arr8)
        # powers2 = [2**i for i in range(7, 7-8, -1)]
        # sum_to_check = sum(x * power for x, power in zip(arr8, powers2))
        # assert byte == sum_to_check, "{} * {} = {}, should be {}".format(arr8, powers2, sum_to_check, byte)
        res += arr8
    return res


if __name__ == "__main__":
    saves_dir = "/home/wesley/.minecraft/saves/"
    world_name = "PythonExperimentation"
    region_dir = os.path.join(saves_dir, world_name, "region")
    region_tup = (0, 0)  # x,z or something (mod 32)
    region_filename = "r.{}.{}.mca".format(*region_tup)
    fp = os.path.join(region_dir, region_filename)
    
    with open(fp, "rb") as f:
        contents = f.read()
    
    # print(len(contents))
    # print(type(contents))
    
    # print(contents[:10])
    # print(contents[:100])
    
    # from bitstring import BitArray
    # import bitstring
    # from bitstring import BitArray
    # c = BitArray(hex=contents[:10])
    assert all(type(x) is int for x in contents)
    assert len(contents) == len(list(contents))

    bit_array = to_bit_array(contents)
    print(len(bit_array))
    print(bit_array[:100])

