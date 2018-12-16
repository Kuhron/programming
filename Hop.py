# idea from chessboard with 4 black pawns and 4 white pawns
# start them in some configuration
# I chose the chessboard to have units of length of 1/2 square side,
# so the board was effectively 16x16, and with toroidal array
# start with some piece, e.g. one closest to center
# or always start a black piece on the center and place others randomly
# find closest piece of opposite color to it
# metric: shortest distance along horizontal, vertical, OR diagonal segments
# if tied for closest, choose randomly
# that piece of opposite color hops away from the reference piece
# in the same direction and with the same magnitude
# then that piece that just hopped (was repelled) becomes the new reference piece
# so it will alternate with black and white pieces moving
# haven't decided yet what happens in collision, maybe masses combine (+ and - will cancel) or maybe they just occupy the same space and pass through each other
# what behavior arises?

# also: what does distance in this metric look like?
# what does a circle look like?


import random
import time

import numpy as np
# import matplotlib.pyplot as plt


class Board:
    def __init__(self, side_length):
        self.side_length = side_length
        self.array = np.zeros((self.side_length, self.side_length))
        self.occupied_points = {}
        self.active_point = (0, 0)

    def populate(self, n_pieces_per_color):
        assert 0 < n_pieces_per_color <= (self.side_length ** 2) / 2
        for i in range(n_pieces_per_color * 2):
            value = (-1) ** i
            if i == 0:
                self.set_piece_at_point((0, 0), 1)
            else:
                point = None
                point_is_occupied = True
                while point_is_occupied:
                    point = (
                        random.randrange(self.side_length),
                        random.randrange(self.side_length),
                    )
                    point_is_occupied = point in self.occupied_points
                self.set_piece_at_point(point, value)

    def set_piece_at_point(self, point, value):
        self.array[point[0], point[1]] = value
        if value != 0:
            self.occupied_points[point] = value
        else:
            # self.occupied_points.remove(point)
            self.occupied_points.pop(point)  # why is it called this

    def get_piece_at_point(self, point):
        return self.array[point[0], point[1]]

    def move_piece(self, point0, point1):
        point1 = self.adjust(point1)
        # if point1 in self.occupied_points:
        #     raise
        new_value = self.get_piece_at_point(point0) + self.get_piece_at_point(point1)  # just combine them
        self.set_piece_at_point(point1, new_value)
        self.set_piece_at_point(point0, 0)

    def adjust(self, p):
        return (
            p[0] % self.side_length,
            p[1] % self.side_length,
        ) 

    @staticmethod
    def measure_distance(p0, p1):
        dx = abs(p0[0] - p1[0])
        dy = abs(p0[1] - p1[1])
        # return dx + dy  # L1; circle is a diamond
        # return (dx ** 2 + dy ** 2) ** 0.5  # L2
        return max(dx, dy)  # L1 with diagonal steps; circle is a square

    def step(self):
        active_color = sign(self.get_piece_at_point(self.active_point))
        opposite_color = -1 * active_color
        points = [p for p, v in self.occupied_points.items() if sign(v) == opposite_color]
        print("active point and color:", self.active_point, active_color)
        print("occupied points:", self.occupied_points)
        closest_point = min(points, key=lambda p: (Board.measure_distance(self.active_point, p), random.random()))
        print("closest point of opposite color:", closest_point)
        displacement_vector = Board.get_displacement_vector(self.active_point, closest_point)
        print("displacement:", displacement_vector)
        # move closest piece away from active point, with same displacement
        # but should adjust for increased mass if two same-colored pieces combined, somehow, while keeping pieces on the grid
        # mass = abs(self.get_piece_at_point(closest_point))  # just ignore this for now
        target_point = (
            closest_point[0] + displacement_vector[0],
            closest_point[1] + displacement_vector[1],
        )
        target_point = self.adjust(target_point)
        print("target_point:", target_point, "with value", self.get_piece_at_point(target_point))
        self.move_piece(closest_point, target_point)
        resultant_value = self.get_piece_at_point(target_point)
        print("resultant value at target:", resultant_value)
        if resultant_value == 0:
            # annihilation occurred
            print("annihilation occurred")
            if self.is_empty():
                print("entire board has been annihilated")
                return
            self.active_point = random.choice([k for k in self.occupied_points])
        else:
            self.active_point = target_point

    @staticmethod
    def get_displacement_vector(p0, p1):
        return (p1[0] - p0[0], p1[1] - p0[1])

    def is_empty(self):
        return len(self.occupied_points) == 0
    
    def print(self):
        s = ""
        for row in self.array:
            for item in row:
                sg = " +-"[sign(item)]
                value = int(abs(item))
                sv = str(value) if value != 0 else " "
                assert value < 10
                s += sg + sv + " "
            s += "\n"
        print(s)


def sign(item):
    return 1 if item > 0 else -1 if item < 0 else 0


if __name__ == "__main__":
    board = Board(16)
    board.populate(12)
    board.print()

    for i in range(100):
        print(i)
        board.step()
        board.print()
        # input("press enter to continue\n")
        time.sleep(0.1)
        if board.is_empty():
            break
