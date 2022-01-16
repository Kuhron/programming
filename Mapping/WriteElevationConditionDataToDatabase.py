from IcosahedronPointDatabase import IcosahedronPointDatabase
from LoadMapData import get_condition_shorthand_dict
from ImageMetadata import get_image_metadata_dict
import os


def int_array_file_to_array(fp):
    with open(fp) as f:
        lines = f.readlines()
    arr = []
    for l in lines:
        l = l.strip().split(",")
        row = [int(x) if x != "" else None for x in l]
        arr.append(row)
    return arr


def get_point_number_to_condition_value_dict(pixel_to_icosa_fp, condition_array_fp):
    point_number_arr = int_array_file_to_array(pixel_to_icosa_fp)
    condition_arr = int_array_file_to_array(condition_array_fp)
    nrows1 = len(point_number_arr)
    nrows2 = len(condition_arr)
    assert nrows1 == nrows2
    nrows = nrows1
    ncols1 = len(point_number_arr[0])
    for row in point_number_arr:
        assert len(row) == ncols1
    ncols2 = len(condition_arr[0])
    for row in condition_arr:
        assert len(row) == ncols2
    assert ncols1 == ncols2
    ncols = ncols1

    # okay now we have them and they have the same size, so make the dict
    d = {}
    for row_i in range(nrows):
        pn_row = point_number_arr[row_i]
        cond_row = condition_arr[row_i]
        for pn, cond in zip(pn_row, cond_row):
            assert pn not in d, pn
            if cond is not None:
                d[pn] = cond
                # if it's None then this point is unspecified for that condition, so don't waste memory on it
    return d


if __name__ == "__main__":
    var_to_write = "elevation_condition"
    root_dir = "/home/wesley/Desktop/Construction/Conworlding/Cada World/Maps/CadaIIMapData/"
    if IcosahedronPointDatabase.db_exists(root_dir):
        db = IcosahedronPointDatabase.load(root_dir)
        print("db loaded")
    else:
        block_size = 2**20
        db = IcosahedronPointDatabase.new(root_dir, block_size)
        db.add_variable(var_to_write)
        # don't do elevation in this script, just is_land
        print("new db created")

    # get the land/sea values from images at some icosahedron point resolution
    metadata = get_image_metadata_dict()
    for image_name in metadata.keys():
        print(image_name)
        pixel_to_icosa_fp = metadata[image_name]["pixel_to_icosa_fp"]
        condition_array_dir = metadata[image_name]["condition_array_dir"]
        variable_value_key_fp = os.path.join(condition_array_dir, f"ImageKey_{var_to_write}.csv")
        condition_array_fp = os.path.join(condition_array_dir, f"{image_name}_{var_to_write}_shorthand.txt")

        pn_to_cond = get_point_number_to_condition_value_dict(pixel_to_icosa_fp, condition_array_fp)
        print("array size:", len(pn_to_cond))
        for pn, cond in pn_to_cond.items():
            db[pn, var_to_write] = cond

        # done with this image so save the db
        db.write(clear_cache=True)
    print("done")
