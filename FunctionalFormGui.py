import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_function(x_min, x_max, y_min, y_max):
    im = Image.open("FunctionalFormGuiInput.png")
    pix = im.load()
    assert pix[0, 0] in [(0, 0, 0, 255), (255, 255, 255, 255)]

    x_transform = lambda x: x_min + (x - 0)/(im.size[0] - 0) * (x_max - x_min)
    y_transform = lambda y_: (lambda y: y_min + (y - 0)/(im.size[1] - 0) * (y_max - y_min))(im.size[1] - y_)

    f_dict = {}
    for x in range(im.size[0]):
        vals = [y_transform(y) for y in range(im.size[1]) if pix[x, y] != (255, 255, 255, 255)]
        f_dict[x_transform(x)] = np.mean(vals) if vals != [] else 0

    def f(x):
        if x in f_dict:
            return f_dict[x]
        keys = f_dict.keys()
        if x >= max(keys):
            return f_dict[max(keys)]
        if x <= min(keys):
            return f_dict[min(keys)]

        lo = max(i for i in keys if i <= x)
        hi = min(i for i in keys if i >= x)

        return f_dict[lo] + (x - lo)/(hi - lo) * (f_dict[hi] - f_dict[lo])

    return f


def get_distribution_function(xs, weight_function):
    ws = [weight_function(x) for x in xs]
    s = sum(ws)
    norm_ws = [w/s for w in ws]
    return lambda: np.random.choice(xs, p=norm_ws)


if __name__ == "__main__":
    f = get_function(-100, 100, 0, 1)
    xs = range(-100, 101)
    ys = [f(x) for x in xs]
    # print("\n".join([str(i) for i in zip(xs, ys)]))
    plt.plot(xs, ys)
    plt.show()

    dist = get_distribution_function(xs, f)
    samples = [dist() for i in range(10000)]
    plt.hist(samples, bins=30)
    plt.show()