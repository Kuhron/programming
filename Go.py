# implement basic rules of Go including ko (no superko)
# mess with stuff to see what patterns emerge

# ideas:
# - order points of the board in various ways (interesting patterns, or at random), and have players place stones at the points in order (looping around once all have been done), if one is illegal, try the next one, etc. until nothing is possible for the current player; see how the patterns evolve
# - same thing but with toroidal array liberties


import os
import random
import time


class IllegalMoveException(Exception):
    pass


class Board:
    OPEN = 0
    BLACK = 1
    WHITE = 2

    CHARS = " +O"

    def __init__(self, size=19):
        self.size = size
        self.grid = [[Board.OPEN for j in range(size)] for i in range(size)]
        self.points = [(i, j) for j in range(size) for i in range(size)]
        self.prohibited_ko_point = None

    def add(self, color, position):
        assert color in [Board.BLACK, Board.WHITE]
        if self.get_state_at(position) == Board.OPEN:
            self.set_state_at(position, color)
        self.remove_captured_chains(position)

    def get_state_at(self, position):
        return self.grid[position[0]][position[1]]

    def set_state_at(self, position, new_state):
        self.grid[position[0]][position[1]] = new_state

    def play_at(self, position, color):
        if self.move_is_suicidal(position, color) or position == self.prohibited_ko_point or self.get_state_at(position) != Board.OPEN:
            raise IllegalMoveException
        self.set_state_at(position, color)
        self.remove_captured_chains(position)
        position_just_captured = Exception  # TODO
        if self.move_is_ko(position_just_captured, color):
            self.prohibited_ko_point = position_just_captured
        else:
            self.prohibited_ko_point = None

    def move_is_suicidal(self, position, color):
        # TODO
        # return False
        return self.count_chain_liberties(position, simulated_color_at_position=color) == 0

    def move_is_ko(self, position, color_just_played):
        # TODO
        return False

    @staticmethod
    def get_opposite_color(color):
        assert color in [Board.BLACK, Board.WHITE]
        return Board.WHITE if color == Board.BLACK else Board.BLACK

    def remove_captured_chains(self, most_recent_move_position):
        p = most_recent_move_position
        color = self.get_state_at(p)
        # neighbors_of_opposite_color = [p_ for p_ in self.get_neighbors(p) if self.get_state_at(p_) == get_opposite_color(color)]
        filled_neighbors = [p_ for p_ in self.get_neighbors(p) if self.get_state_at(p_) != Board.OPEN]
        to_remove = set()
        for p in filled_neighbors:
            if self.count_chain_liberties(p) == 0:
                to_remove |= self.get_chain_members(p)
        for p in to_remove:
            self.set_state_at(p, Board.OPEN)

    def count_chain_liberties(self, starting_position, simulated_color_at_position=None):
        return len(self.get_chain_liberties(starting_position, simulated_color_at_position))

    def get_chain_members(self, starting_position, simulated_color_at_position=None, points_already_checked=None):
        color = self.get_state_at(starting_position) if simulated_color_at_position is None else simulated_color_at_position
        if points_already_checked is None:
            points_already_checked = set()
        points_already_checked.add(starting_position)
        neighbors = self.get_neighbors(starting_position)
        same_colors = [p for p in neighbors if self.get_state_at(p) == color]
        to_check = [p for p in same_colors if p not in points_already_checked]
        return set().union({starting_position}, *[self.get_chain_members(p, points_already_checked) for p in to_check])

    def get_chain_liberties(self, starting_position, simulated_color_at_position=None, points_already_checked=None):
        # need to implement this using get_chain_members as auxiliary function
        color = self.get_state_at(starting_position) if simulated_color_at_position is None else simulated_color_at_position
        if points_already_checked is None:
            points_already_checked = set()
        points_already_checked.add(starting_position)
        neighbors = self.get_neighbors(starting_position)
        liberties = {p for p in neighbors if self.get_state_at(p) == Board.OPEN}
        same_colors = [p for p in neighbors if self.get_state_at(p) == color]
        to_check = [p for p in same_colors if p not in points_already_checked]
        return set().union(liberties, *[self.get_chain_liberties(p, points_already_checked) for p in to_check])

    def get_neighbors(self, starting_position):
        n = self.size
        is_valid = lambda p: 0 <= p[0] < n and 0 <= p[1] < n
        a, b = starting_position
        return [x for x in [(a+1, b), (a-1, b), (a, b+1), (a, b-1)] if is_valid(x)]

    def print(self):
        s = "\n"*100  # janky os.system("clear") with hopefully no jumpiness
        s += "/" + "-" * self.size + "\\\n"
        for i in range(self.size):
            s += "|" + " ".join(Board.CHARS[self.get_state_at((i, j))] for j in range(self.size)) + "|\n"
        s += "\\" + "-" * self.size + "/\n"
        print(s)


def play_points_in_order(board, ordering):
    # n = len(ordering) ** 0.5
    # assert n < 100, "board too big"
    # assert abs(n - int(n)) < 1e-6, "ordering contains {} elements, but this should be a perfect square".format(len(ordering))
    # board = Board(int(n))
    player = Board.BLACK
    point_index = 0
    last_legal_point_index = None
    sleep = True
    while True:
        board.print()
        point_to_play_at = ordering[point_index]
        if point_index == last_legal_point_index:
            # all points were tried without being able to play
            break
        try:
            board.play_at(point_to_play_at, player)
            player = Board.get_opposite_color(player)
            last_legal_point_index = point_index
            if sleep:
                time.sleep(0.01)
        except IllegalMoveException:
            pass

        point_index = (point_index + 1) % len(ordering)
        if point_index == 0:
            sleep = False


if __name__ == "__main__":
    n = 9
    board = Board(n)
    ordering = [x for x in board.points]  # default order
    # ordering = sorted(ordering, key=lambda x: random.random())  # random order
    ordering = sorted(ordering, key=lambda x: x[0] + (x[1] * (1 + 1e-9)))  # diagonal order, picking a directional ordering along each diagonal, hence the (1 + 1e-9)
    play_points_in_order(board, ordering)

