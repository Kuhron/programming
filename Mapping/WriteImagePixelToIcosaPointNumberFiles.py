# this script should just call functions that exist elsewhere,
# don't import this, just use it to generate the files once
# (or again if they are deleted / need to be changed)

from TransformImageIntoMapData import write_image_pixel_to_icosa_point_number
from ImageMetadata import get_image_metadata_dict


if __name__ == "__main__":
    metadata = get_image_metadata_dict()
    # image_names_to_use = ["Legron"]
    image_names_to_use = sorted(metadata.keys())
    # image_names_already_done = ["Sertorisun Islands", "Legron", "Imis Tolin", "Mienta"]
    image_names_already_done = []

    for name in image_names_already_done:
        image_names_to_use.remove(name)
    for image_name in image_names_to_use:
        write_image_pixel_to_icosa_point_number(image_name)
