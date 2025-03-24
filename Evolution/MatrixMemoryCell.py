# trying to make a simple structure that can "remember" things and react to environmental stimuli
# represent it as a square matrix, restrict all entries to single digits for ease of reading


import math
import random
import numpy as np
from collections import defaultdict

# terminal color printing
from colorama import init as colorama_init
from colorama import Fore, Back, Style
colorama_init()


pi = np.pi

BOX_CHARS = {
    "top-left": "\u2554",
    "top-right": "\u2557",
    "bottom-left": "\u255a",
    "bottom-right": "\u255d",
    "vertical": "\u2551",
    "horizontal": "\u2550",
}

CHARS = "0123456789XJQKRV"
COLORS = {
    0: Fore.WHITE, 1: Fore.RED, 2: Fore.BLUE, 3: Fore.GREEN, 4: Fore.YELLOW, 5: Fore.MAGENTA, 6: Fore.CYAN, 7: Fore.LIGHTBLACK_EX,
    8: Fore.BLACK+Back.WHITE, 9: Fore.BLACK+Back.RED, 10: Fore.BLACK+Back.BLUE, 11: Fore.BLACK+Back.GREEN, 12: Fore.BLACK+Back.YELLOW, 13: Fore.BLACK+Back.MAGENTA, 14: Fore.BLACK+Back.CYAN, 15: Fore.BLACK+Back.LIGHTBLACK_EX,
}
INT_TO_CHAR = dict(enumerate(CHARS))
INT_TO_COLOR = COLORS
INT_MOD = len(INT_TO_CHAR)
DIRECTION_ORDER = ["right", "down", "left", "up"]


class Cell:
    def __init__(self, n: int):
        assert 1 <= n
        self.n = n
        self.array = np.zeros((n,n), dtype=int)
        self.size = Cell.get_entry_size(n)
        self.border_size = Cell.get_border_size(n)

    def __repr__(self):
        top_row = BOX_CHARS["top-left"] + (BOX_CHARS["horizontal"]) * (2 * self.n + 1) + BOX_CHARS["top-right"]
        rows = []
        for i in range(self.n):
            row = BOX_CHARS["vertical"] + " " + " ".join(Cell.str_from_int(x) for x in self.array[i]) + " " + BOX_CHARS["vertical"]
            rows.append(row)
        bottom_row = BOX_CHARS["bottom-left"] + (BOX_CHARS["horizontal"]) * (2 * self.n + 1) + BOX_CHARS["bottom-right"]
        s = top_row + "\n" + "\n".join(rows) + "\n" + bottom_row
        return s

    @staticmethod
    def str_from_int(x):
        return INT_TO_COLOR[x] + INT_TO_CHAR[x] + Style.RESET_ALL

    def get_entry_index(self, i):
        r,c = Cell.entry_index_to_row_column(i, self.n)
        return self.array[r,c]

    def set_entry_index(self, i, val):
        r,c = Cell.entry_index_to_row_column(i, self.n)
        self.array[r,c] = val

    def get_border_index(self, i):
        r,c = Cell.border_index_to_row_column(i, self.n)
        return self.array[r,c]

    def set_border_index(self, i, val):
        r,c = Cell.border_index_to_row_column(i, self.n)
        self.array[r,c] = val

    @staticmethod
    def get_entry_size(n):
        return n**2

    @staticmethod
    def get_border_size(n):
        return 4 * (n - 1)

    @staticmethod
    def entry_index_to_row_column(i: int, n: int):
        if i < 0 or i >= Cell.get_entry_size(n):
            raise IndexError(i)
        return divmod(i, n)

    @staticmethod
    def border_index_to_row_column(i: int, n: int):
        if i < 0 or i >= Cell.get_border_size(n):
            raise IndexError(i)
        side, index_in_side = divmod(i, n - 1)
        if side == 0:
            r = 0
            c = i
        elif side == 1:
            r = index_in_side
            c = n - 1
        elif side == 2:
            r = n - 1
            c = n - 1 - index_in_side
        elif side == 3:
            r = n - 1 - index_in_side
            c = 0
        else:
            raise ValueError(f"shouldn't happen: {side = }")
        return r,c

    def randomize(self, seed):
        r = rgen(seed)
        for i in range(self.size):
            self.set_entry_index(i, Cell.int_from_rand(next(r)))

    @staticmethod
    def int_from_rand(x):
        return int(INT_MOD * x)

    @staticmethod
    def sin_from_int(x):
        if x == 0 or x == INT_MOD/2:
            return 0.0  # float crap at sin(pi)
        return np.sin(2*pi * x/INT_MOD)

    @staticmethod
    def add(x, y):
        return int((x + y) % INT_MOD)

    def get_border(self):
        return [self.get_border_index(i) for i in range(self.border_size)]

    def connection_strength(self, rc1, rc2):
        r1, c1 = rc1
        r2, c2 = rc2
        x1 = self.array[r1, c1]
        x2 = self.array[r2, c2]
        x = Cell.add(x1, x2)
        strength = Cell.sin_from_int(x)
        return strength

    def poke(self, border_index):
        mag = 1
        rc = Cell.border_index_to_row_column(border_index, self.n)
        impulses_to_send = {rc: mag}

        while len(impulses_to_send) > 0:
            impulses_to_send = self.receive_impulses(impulses_to_send)

    def receive_impulse(self, magnitude, rc):
        # the impulse is added to the value at (r,c)
        # then iterate over the neighbors of this array element, sending the same impulse size times the connection strength

        debug = False
        if debug:
            print(f"\nbefore receiving impulse {magnitude} at {rc}:\n")
            print(self)

        r,c = rc
        x = self.array[r,c]
        new_x = Cell.add(x, magnitude)
        self.array[r,c] = new_x

        if debug:
            print(f"\nafter impulse {magnitude} at {rc}:\n")
            print(self)
            input("check")

        impulses_to_send = defaultdict(float)
        if new_x != x:
            # now add the new impulses that will be given to neighbors
            for direction in DIRECTION_ORDER:
                rc2 = self.neighbor(rc, direction)
                if rc2 is None:  # neighbor does not exist (off edge of cell)
                    continue
                strength = self.connection_strength(rc, rc2)
                mag_to_send = magnitude * strength
                if mag_to_send != 0:
                    impulses_to_send[rc2] += mag_to_send

        # return all resultant impulses from this single received impulse, they will be aggregated later
        return impulses_to_send

    def receive_impulses(self, impulses_to_send):
        new_impulses_to_send = defaultdict(float)
        for rc, mag in impulses_to_send.items():
            for rc2, mag2 in self.receive_impulse(mag, rc).items():
                new_impulses_to_send[rc2] += mag2

        # here, aggregate the resultant impulses from the received impulses, inhibition will cause positive impulses not to be sent if it outweighs them
        return with_only_positive_values(new_impulses_to_send)

    def neighbor(self, rc, direction):
        r,c = rc
        if direction == "right":
            if c == self.n-1:
                return None
            return r, c+1
        elif direction == "down":
            if r == self.n-1:
                return None
            return r+1, c
        elif direction == "left":
            if c == 0:
                return None
            return r, c-1
        elif direction == "up":
            if r == 0:
                return None
            return r-1, c
        else:
            raise ValueError(f"invalid direction: {direction}")


def without_zero_values(d):
    return {k:v for k,v in d.items() if v != 0}


def with_only_positive_values(d):
    return {k:v for k,v in d.items() if v > 0}


def rgen(seed):
    # create local instance of random.Random RNG so it's thread-safe (other invocations of random functions elsewhere won't advance the RNG state)
    r = random.Random()
    r.seed(seed)
    while True:
        yield r.random()



if __name__ == "__main__":
    cell = Cell(39)
    # cell.randomize("cataract32")
    print(cell)
    # print(cell.get_border())

    i = 0
    last_i_printed = 0
    r = rgen("cataract32")
    while True:
        index = int(cell.border_size * next(r))
        cell.poke(index)
        if i - last_i_printed >= 1e5 and random.random() < 1e-5:
            print(i)
            print(cell)
            last_i_printed = i
        i += 1
