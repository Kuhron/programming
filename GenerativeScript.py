import random
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def plot_bezier_curve(vertices, lower_left_corner_coords):
    # https://matplotlib.org/gallery/api/quad_bezier.html
    Path = mpath.Path
    codes = []
    codes.append(Path.MOVETO)  # first one is always this, put the pen at this point
    n = len(vertices)
    if n == 2:
        codes.append(Path.LINETO)  # line segment
    elif n == 3:
        codes += [Path.CURVE3] * 2  # quadratic spline
    elif n == 4:
        codes += [Path.CURVE4] * 3  # cubic spline
    else:
        raise ValueError("invalid number of vertices: {}".format(len(vertices)))
    # transform coordinates of vertices to be in the unit box with specified lower left corner
    # scale the char down as well by 0.9 or something (toward center of box)
    llx, lly = lower_left_corner_coords
    cx, cy = llx + 0.5, lly + 0.5
    def transform_vertex(v):
        vx, vy = v
        dx, dy = vx - 0.5, vy - 0.5
        scale = 1
        dx *= scale
        dy *= scale
        return (cx + dx, cy + dy)
    vertices = [transform_vertex(v) for v in vertices]
    pp = mpatches.PathPatch(Path(vertices, codes), fill=False)
    plt.gca().add_patch(pp)


def plot_random_glyph(lower_left_corner_coords):
    segments = get_random_glyph_shape()
    plot_glyph(segments, lower_left_corner_coords)


def plot_glyph(segments, lower_left_corner_coords):
    for vertices in segments:
        plot_bezier_curve(vertices, lower_left_corner_coords)


def get_random_glyph_shape():
    segments = []
    n_segments = random.randint(2, 6)
    points_of_interest = set()  # can draw from these points sometimes instead of just random points

    def get_point():
        if len(points_of_interest) != 0 and random.random() < 0.7:
            return random.choice(list(points_of_interest))
        return tuple(np.random.rand(2))

    def get_point_set(n_points):
        res = []
        for i in range(n_points):
            p = get_point()
            while p in res:
                p = get_point()
            res.append(p)
        return res

    for i in range(n_segments):
        n_vertices = random.randint(2, 4)
        # vertices = [np.random.rand(2) for j in range(n_vertices)]
        vertices = get_point_set(n_vertices)
        segments.append(vertices)

        points_of_interest |= set(vertices)
        # get center of mass or something like that
        com = sum(np.array(p) for p in vertices) / len(vertices)
        com = tuple(com)
        points_of_interest.add(com)

    return segments


def get_random_alphabet():
    # skew distribution of letter numbers to smaller
    x = random.uniform(np.sqrt(12), np.sqrt(100))
    n_glyphs = int(x**2)
    alphabet = [get_random_glyph_shape() for i in range(n_glyphs)]
    return alphabet


def get_random_word(alphabet):
    word_len = random.randint(1, 20)
    glyph_numbers = [random.randrange(len(alphabet)) for j in range(word_len)]
    glyphs = [alphabet[g] for g in glyph_numbers]
    return glyphs


def plot_random_text(alphabet):
    words = []
    n_words = random.randint(30, 100)
    for i in range(n_words):
        words.append(get_random_word(alphabet))

    # select writing direction (8 possibilities)
    switch = lambda: 1 if random.random() < 0.5 else -1
    char_step, line_step = (np.array([0, 1]), np.array([1, 0]))[::switch()]
    char_step *= switch()
    line_step *= switch()

    plot_text(words, char_step, line_step)


def plot_alphabet(alphabet):
    n = len(alphabet)
    side_len = int(np.ceil(np.sqrt(n)))
    words = []
    i = 0
    for line_i in range(side_len):
        word = alphabet[side_len * line_i : side_len * (line_i + 1)]
        words.append(word)
    # for showing alphabet, just use English direction
    char_step = np.array([1, 0])
    line_step = np.array([0, -1])
    plot_text(words, char_step, line_step)


def plot_text(words, char_step, line_step):    
    starting_llc = np.array([0, 0])  # lower-left corner of box for glyph
    def get_llc(line_number, char_number):
        return starting_llc + (line_number * line_step) + (char_number * char_step)

    line_len = max(len(w) for w in words)
    line_number = 0
    char_number = 0
    for w in words:
        n = len(w)
        if char_number + n > line_len:
            # carriage return
            line_number += 1
            char_number = 0
        if char_number == 0:
            assert n <= line_len, "line len can't accommodate word of len {}".format(n)
        for glyph in w:
            llc = get_llc(line_number, char_number)
            plot_glyph(glyph, llc)
            char_number += 1
        char_number += 1 # space
        

    plt.gca().relim()      # make sure all the data fits
    plt.gca().autoscale()  # auto-scale
    plt.gca().set_aspect("equal")



if __name__ == "__main__":
    alphabet = get_random_alphabet()
    plot_alphabet(alphabet)
    plt.show()
    plot_random_text(alphabet)
    plt.show()
