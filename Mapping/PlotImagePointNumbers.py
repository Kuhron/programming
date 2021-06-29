# just for fun, to see what structure may exist
# in the pattern of icosa point numbers for an image on the map

import random
import numpy as np
import matplotlib.pyplot as plt

from ImageMetadata import get_image_metadata_dict


if __name__ == "__main__":
    metadata = get_image_metadata_dict()
    image_names = sorted(metadata.keys())
    image_name = random.choice(image_names)
    point_number_fp = metadata[image_name]["pixel_to_icosa_fp"]

    with open(point_number_fp) as f:
        lines = f.readlines()
    strs = [l.strip().split(",") for l in lines]
    ints = [[int(x) for x in s] for s in strs]

    modulus = 2 ** random.randint(4, 30)
    arr = np.mod(ints, modulus)
    plt.imshow(arr)
    plt.title(f"{image_name} mod {modulus}")
    plt.show()
    
