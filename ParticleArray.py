# inspired by attempts to create a zero-player game involving laying down one playing card at a time onto a surface
# each card is only distinguished by suit and has two parameters: color (+ = S, C; - = H, D) and majority (+ = S, H; - = D, C)
# lay down one card at a time, starting with the first card in the center
# each new card starts where the last card settled, and is subject to a force of one unit in a certain direction
# same-color force = northeast, same-majority force = southeast; adding these yields:
# same suit = east, same color = north, same majority = south, opposite suit (neither parameter matches) = west
# the card is moved one unit in that direction and then placed, and its parameters are added to the card that is already there, if any
# adding same suits has no effect; adding opposite suits annihilates both cards
# adding same color but opposite majority creates a "color group" which will exert a northeast or southwest force on new cards
# adding same majority but opposite color creates a "majority group" which will exert a southeast or northwest force on new cards
# a diagonal force results in a move of one unit along each axis, without interacting with cards that it would pass over on the way there
# adding a single card to a combined group has the same effect as just adding the effects of each stacked card in that spot
# (it should be true that the combined groups previously named are just additive like this as well)
# addition table for parameters (add each parameter separately to get the new parameter value for the resultant group):
#    -1  0 +1
# -1 -1 -1  0
#  0 -1  0 +1
# +1  0 +1 +1
# could make things accumulate instead of doing this, and just add all their force vectors without bound, as an alternative

# so there are four elementary particles
# could think of these as the suits, or as + and - particles for each parameter, with 0 as the other parameter
# either way works, since suits can be composed of parameter particles and parameter particles (combined groups) can be composed of suits
# but the distinction is not arbitrary since the only things that "arrive" and are subject to force are suits, not combined groups
# there are 9 possible values for each cell in the array (unless decide to let parameters add as normal integers)


# should probably have used numpy arrays from the get-go for clarity, oh well. TODO

# problems:
# - not deterministic given initial state
# - basically no persistent zero particles


import math
import random
import time


def add_values(a, b, restrict=True):
    if restrict:
        valid_values = [-1, 0, 1]
        assert a in valid_values and b in valid_values
        if a == 0:
            return b
        if b == 0:
            return a
        if a == b:
            return a
        if a == -b:
            return 0
        raise Exception("should not get here")
    else:
        return a + b


def create_array(side_length):
    func = create_zero_particle
    # func = create_random_particle
    return [[func() for col in range(side_length)] for row in range(side_length)]


def create_random_particle():
    return (random.choice([-1, 1]), random.choice([-1, 1]))


def create_zero_particle():
    return (0, 0)


def create_cyclical_particle(i):
    return [(1, 1), (-1, 1), (-1, -1), (1, -1)][i % 4]


def evolve_array(arr, last_position, i):
    n_rows = len(arr)
    n_cols = len(arr[0])
    initial_position = last_position
    last_row, last_col = last_position
    # new_particle = create_random_particle()  # but prefer it to be deterministic for a given initial state
    # new_particle = arr[last_row][last_col]  # but once a zero is created, evolution will cease
    new_particle = create_cyclical_particle(i)  # returns suits in cyclical order
    visited_positions = []  # annihilate all particles in a loop if one occurs
    annihilate = False
    while True:
        existing_particle = arr[last_row][last_col]
        displacement = get_displacement(existing_particle, new_particle)
        if displacement == (0, 0):
            break
        last_row = (last_row + displacement[0]) % n_rows
        last_col = (last_col + displacement[1]) % n_cols
        last_position = (last_row, last_col)
        if last_position in visited_positions:
            annihilate = True
            break  # will annihilate the loop; do not place position's re-occurrence in the list
        visited_positions.append(last_position)

    if annihilate:
        # delete new particle, i.e., don't give it any effect on others or place it anywhere
        loop_boundary = last_position
        index = [i for i, x in enumerate(visited_positions) if x == loop_boundary][-1]  # the second occurrence was not placed in list
        to_annihilate = visited_positions[index:]
        for row, col in to_annihilate:
            arr[row][col] = (0, 0)
        return arr, initial_position  # don't use last position since it is at an indeterminate place in the loop
    else:
        landing_particle = arr[last_row][last_col]
        color = add_values(landing_particle[0], new_particle[0])
        majority = add_values(landing_particle[1], new_particle[1])
        arr[last_row][last_col] = (color, majority)
        return arr, last_position
        # return arr, initial_position  # always insert at the origin


def get_displacement(p1, p2):
    c1, m1 = p1
    c2, m2 = p2
    color_force = (-1, 1)  # northeast, i.e., decreasing row and increasing column
    majority_force = (1, 1)  # southeast, i.e., increasing row and increasing column
    color_contribution = c1 * c2
    majority_contribution = m1 * m2
    return (
        add_values(color_contribution * color_force[0], majority_contribution * majority_force[0]),
        add_values(color_contribution * color_force[1], majority_contribution * majority_force[1]),
    )


def params_to_char(tup):
    # characters indicate what direction a (+1, +1), a spade, would be pushed
    color, majority = tup
    color = -1 if color < 0 else 1 if color > 0 else 0
    majority = -1 if majority < 0 else 1 if majority > 0 else 0

    if color == -1:  # red
        return "←" if majority == -1 else "↙" if majority == 0 else "↓" if majority == 1 else None
    elif color == 0:
        return "↖" if majority == -1 else "-" if majority == 0 else "↘" if majority == 1 else None
    elif color == 1:  # black
        return "↑" if majority == -1 else "↗" if majority == 0 else "→" if majority == 1 else None
    else:
        raise Exception("should not get here")


def print_array(arr):
    header = "--" * len(arr[0]) + "\n"
    s = header
    for row in arr:
        s += " ".join(params_to_char(tup) for tup in row) + "\n"
    s += header
    print(s)


if __name__ == "__main__":
    side_length = 35
    arr = create_array(side_length)
    last_position = [int((side_length + 1) / 2)] * 2
    i = 0
    while True:
        if i % 1000 == 0:
            print_array(arr)
            print(i)
            time.sleep(0.05)
        arr, last_position = evolve_array(arr, last_position, i)
        i += 1
