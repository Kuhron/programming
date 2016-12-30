import os
import zlib


WORLD_DIR = "C:/Users/Wesley/AppData/Roaming/.minecraft/saves/"
DEFAULT_WORLD = "PythonExperimentation"


def get_region_dir_from_world_name(world):
    return WORLD_DIR + world + "/region/"


if __name__ == "__main__":
    world = DEFAULT_WORLD
    filepath = get_region_dir_from_world_name(world) + "r.0.0.mca"

    with open(filepath, "rb") as f:
        bytes = f.read()

    header_data = bytes[:8192]
    location_data = header_data[:4096]
    timestamp_data = header_data[4096:]
    chunk_data = bytes[8192:]
    assert len(chunk_data) % 4096 == 0

    locations = [location_data[4 * i : 4 * (i + 1)] for i in range(1024)]

    location_offsets = [x[:3] for x in locations]
    raw_sector_indices = [int.from_bytes(x, byteorder="big") for x in location_offsets]
    sector_indices = [x - 2 if x != 0 else None for x in raw_sector_indices]
    assert(len([x for x in sector_indices if x is not None]) == len(set(sector_indices) - {None}))  # assert unique sector offsets

    chunk_lengths = [x[3] for x in locations]
    timestamps = [timestamp_data[4 * i : 4 * (i + 1)] for i in range(1024)]  # who cares

    chunk_data_sectors = [chunk_data[4096 * i : 4096 * (i + 1)] for i in range(int(len(chunk_data) / 4096))]
    sectors = [chunk_data_sectors[i] if i is not None else None for i in sector_indices]
