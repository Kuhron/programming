from PIL import Image
import os
import numpy as np


class CornerSet:
    def __init__(self, top_left, top_right, bottom_left, bottom_right, host_shape):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.host_shape = host_shape
        self.transform = self.get_transformation()

    def get_transformation(self):
        # the transformation that maps (row, column) to its new position given these corners
        # I believe that upper left corner of each pixel is its value, so e.g. the point (7.3, 11.9) is in the pixel (7, 11)
        # and the upper-leftmost pixel of an image is (0, 0), so the real-valued coordinates measure distance from the very top left corner of the image
        # so the very lower right corner (in the limit?) has coordinates exactly equal to host_shape, but no part of the image is at that pixel
        # won't necessarily be a linear transformation since parallel lines are not preserved if we create a trapezoid or reverse one of the sides but not its opposite (twist in the image)
        def transform(r, c):
            # get a point's alpha along the row and column axes
            r00, c00 = self.top_left
            r01, c01 = self.top_right
            r10, c10 = self.bottom_left
            r11, c11 = self.bottom_right
            rt, ct, colors = self.host_shape
            r_alpha = r/rt
            c_alpha = c/ct
            # get the transformed endpoints of the row this point is on
            # left end of the row is r_alpha of the way between top left and bottom left; simil right
            left_end_r = r00 + r_alpha * (r10 - r00)
            left_end_c = c00 + r_alpha * (c10 - c00)
            right_end_r = r01 + r_alpha * (r11 - r01)
            right_end_c = c01 + r_alpha * (c11 - c01)
            pos_r = left_end_r + c_alpha * (right_end_r - left_end_r)
            pos_c = left_end_c + c_alpha * (right_end_c - left_end_c)
            return pos_r, pos_c
        return transform




if __name__ == "__main__":
    image_dir = "Images/Recursion"
    image_fnames = {
        "house": "000deg_20220330_230210.jpg",
        "graffiti": "0deg_20220914_001143.jpg",
        "desk": "000deg_20191221_203951.jpg",
    }

    name_to_arr = {}
    for name, fname in image_fnames.items():
        with Image.open(os.path.join(image_dir, fname)) as f:
            arr = np.array(f)
        name_to_arr[name] = arr

    for name, arr in name_to_arr.items():
        print(f"{name} has shape {arr.shape}")

    # plug-and-play rules
    # it says for each image, convert it to the next generation of itself by following the ordered replacement rules
    # each replacement rule says take these four corners and use that as the place to insert the last generation of some other image (or itself)
    recipes = {
        "house": [
            [[(500, 1000), (500, 2000), (1500, 1000), (1500, 2000)], "graffiti"],
            [[(1500, 2000), (1500, 3000), (2500, 2000), (2500, 3000)], "desk"],
        ],
        "graffiti": [
            [[(500, 1000), (500, 2000), (1500, 1000), (1500, 2000)], "desk"],
            [[(1500, 2000), (1500, 3000), (2500, 2000), (2500, 3000)], "house"],
        ],
        "desk": [
            [[(500, 1000), (500, 2000), (1500, 1000), (1500, 2000)], "house"],
            [[(1500, 2000), (1500, 3000), (2500, 2000), (2500, 3000)], "graffiti"],
        ],
    }

    # follow these recipes until some level of specificity is reached (e.g. all of the replacements are smaller than a pixel in the oldest iteration, can calculate this by "multiplying" the corners to see where the nth-level embedding of something is located, then stop when all the oldest embeddings have all of their corners within the same 1 pixel of area)

    iterations = 5
    for iteration in range(1, iterations+1):
        print(f"{iteration = }")
        new_name_to_arr = {}  # put the modified images here until we're done with this round, so no changes to an earlier image in the queue get passed on to another image if it looks at the already-modified copy of that earlier image
        for host_name in recipes:
            print(f"{host_name = }")
            host_fname = image_fnames[host_name]
            host_arr = name_to_arr[host_name]
            new_host_arr = host_arr.copy()
            host_rt, host_ct, host_colors = new_host_arr.shape
            for rule in recipes[host_name]:
                corners, guest_name = rule
                corner_set = CornerSet(*[np.array(rc) for rc in corners], new_host_arr.shape)
                guest_arr = name_to_arr[guest_name]
                rt, ct, colors = guest_arr.shape
                pixel_outputs = {}
                C, R = np.meshgrid(range(ct), range(rt))
                assert R.shape == (rt, ct), R.shape
                R2, C2 = corner_set.transform(R, C)
                R2 = R2.astype(int)
                C2 = C2.astype(int)
                for r in range(rt):
                    if r % 100 == 0:
                        print(f"{r = }/{rt}", end="\r")
                    for c in range(ct):
                        rgb = guest_arr[r, c]
                        new_r = R2[r, c]
                        new_c = C2[r, c]
                        new_pos = (new_r, new_c)
                        if not (0 <= new_r < host_rt) or not (0 <= new_c < host_ct):
                            # this pixel will be placed outside the host image's boundaries
                            continue
                        if new_pos not in pixel_outputs:
                            pixel_outputs[new_pos] = []
                        pixel_outputs[new_pos].append(rgb)
                print()
                # now average the outputs for the pixels and place them in the host image
                print("placing sub-image")
                for new_rc, rgbs in pixel_outputs.items():
                    new_r, new_c = new_rc
                    new_host_arr[new_r, new_c] = np.mean(rgbs, axis=0)
                print("done placing sub-image")
            # replace the host image object with the new one so it can be used in future iterations
            new_name_to_arr[host_name] = new_host_arr
            # save this iteration to file
            host_fstr, ext = os.path.splitext(host_fname)
            new_fstr = f"{host_fstr}_iteration{iteration}"
            new_fname = new_fstr + ext

            print("saving image to file")
            im = Image.fromarray(new_host_arr)
            im.save(os.path.join(image_dir, new_fname))
            print("done saving image to file")

        name_to_arr = new_name_to_arr  # now update the references to image arrays

