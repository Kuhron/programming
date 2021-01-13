import os
import random


# documentation of the file format:
# https://minecraft.gamepedia.com/Region_file_format
# https://minecraft.gamepedia.com/Anvil_file_format
# the .mca are Anvil format, which has a different coordinate order but otherwise seems to be the same as Region format

# X is east, Y is up, Z is south


def bytes_to_int(byts, big_endian=True):
    res = 0
    for i in range(len(byts)):
        byte_at_i = byts[-(i+1)]  # big-endian means front end is biggest digit place, like decimal numbers are written
        multiplier = 16 ** i
        res += byte_at_i * multiplier
    return res


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


def get_region_xz(point_xz):
    chunk_x, chunk_z = point_xz
    region_x = chunk_x // 32
    region_z = chunk_z // 32
    return region_x, region_z


def get_offset_in_region_file(point_xz):
    x, z = point_xz
    return 4 * ((x % 32) + (z % 32) * 32)


def get_timestamp_offset_in_region_file(point_xz):
    return 4096 + get_offset_in_region_file(point_xz)


def get_block_at(byts, x, y, z):
    print("getting block at {} {} {}".format(x,y,z))
    offset = get_offset_in_region_file([x, z])
    print("offset is {}".format(offset))
    four_bytes = list(byts[offset: offset+4])
    print("bytes are {}".format(four_bytes))
    offset_4kib = four_bytes[:3]  # in 4 KiB units from start of file (the first 4096 B is chunk location info where these bytes themselves are stored, the second 4096 B are the chunk timestamps, so offset of 2 * 4KiB begins right after the timestamps)
    sector_count = four_bytes[3]  # also expressed as multiples of 4 KiB (sector size)
    offset_4kib_int = bytes_to_int(offset_4kib)
    chunk_data_offset = offset_4kib_int * 4096
    # I don't care about the timestamps, which are just last modification time (although maybe I should update them? idk)
    chunk_data_len = sector_count * 4096
    chunk_data = byts[chunk_data_offset : chunk_data_offset + chunk_data_len]

    data_len_in_bytes_arr = chunk_data[:4]
    data_len_in_bytes = bytes_to_int(data_len_in_bytes_arr)
    compression_type = chunk_data[4]
    data = chunk_data[5:]
    print("data_len_in_bytes {}, compression_type {}".format(data_len_in_bytes, compression_type))
    assert compression_type in [1,2,3], compression_type

    raise NotImplementedError


def set_block_at(byts, x, y, z, block_type):
    raise NotImplementedError



if __name__ == "__main__":
    saves_dir = "/home/wesley/.minecraft/saves/"
    world_name = "PythonExperimentation"
    region_dir = os.path.join(saves_dir, world_name, "region")
    region_tup = (0, 0)  # x,z or something (mod 32)
    region_filename = "r.{}.{}.mca".format(*region_tup)
    fp = os.path.join(region_dir, region_filename)
    
    with open(fp, "rb") as f:
        contents = f.read()
    
    assert all(type(x) is int for x in contents)
    assert len(contents) == len(list(contents))

    # bit_array = to_bit_array(contents)
    # print(len(bit_array))
    # print(bit_array[:100])

    assert get_block_at(contents, 2, 62, -11) == "sand"
    assert get_block_at(contents, 2, 63, -14) == "cactus"
    assert get_block_at(contents, -6, 56, -22) == "iron_ore"

    for y in range(63, 100):
        contents = set_block_at(contents, 4, y, 21, "gold")

    # write_contents_to_region_file(contents, fp)
