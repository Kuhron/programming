import colordict


def get_rgbs_xkcd():
    with open("xkcd_colors_rgb.txt") as f:
        lines = f.readlines()
    d = {}
    for l in lines:
        if l.startswith("#"):
            continue
        color, hx = l.strip().split("\t")
        rgb = colordict.hex_to_rgb(hx)
        d[color] = rgb
    return d


if __name__ == "__main__":
    colors = get_rgbs_xkcd()
    for k, v in colors.items():
        if "ugly" in k:
            print(k, v)
    print(len(colors))
