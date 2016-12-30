import random

from collections import Counter
from copy import deepcopy


class CubePiece:
	def __init__(self, coordinates, n_on_side):
		self.coordinates = coordinates
		self.n_on_side = n_on_side
		self.colors = {0: {-1: "r", 1: "o"}, 1: {-1: "b", 1: "g"}, 2: {-1: "y", 1: "w"}}

	def rotate(self, rotation_axis, rhr_direction):
		if self.is_visible():
			self.rotate_coordinates(rotation_axis, rhr_direction)
			self.rotate_colors(rotation_axis, rhr_direction)

	def rotate_coordinates(self, rotation_axis, rhr_direction):
		new_coordinates = [None, None, None]
		new_coordinates[rotation_axis] = self.coordinates[rotation_axis]

		# rot 0 +1: (0, 0, 3) -> (0, 0, 0); (0, 1, 3) -> (0, 0, 1); (0, 2, 3) -> (0, 0, 2); (0, 3, 3) -> (0, 0, 3)
		# rot 1 +1: (0, 0, 3) -> (3, 0, 3); (1, 0, 3) -> (3, 0, 2); (2, 0, 3) -> (3, 0, 1); (3, 0, 3) -> (3, 0, 0)

		# rot 0 -1: (0, 0, 0) -> (0, 0, 3); (0, 0, 1) -> (0, 1, 3); (0, 0, 2) -> (0, 2, 3); (0, 0, 3) -> (0, 3, 3)

		new_coordinates[(rotation_axis - rhr_direction) % 3] = self.coordinates[(rotation_axis + rhr_direction) % 3]
		new_coordinates[(rotation_axis + rhr_direction) % 3] = (self.n_on_side - 1) - self.coordinates[(rotation_axis - rhr_direction) % 3]

		self.coordinates = tuple(new_coordinates)

	def rotate_colors(self, rotation_axis, rhr_direction):
		new_colors = {rotation_axis: self.colors[rotation_axis]}

		# rot 0 +1: ax 1 +1 -> ax 2 +1
		# rot 1 +1: ax 2 +1 -> ax 0 +1
		# rot 2 +1: ax 0 +1 -> ax 1 +1

		# rot 0 -1: ax 2 +1 -> ax 1 +1

		# rot 0 +1: ax 2 -1 -> ax 1 +1
		# rot 1 +1: ax 0 -1 -> ax 2 +1
		# rot 2 +1: ax 1 -1 -> ax 0 +1

		# rot 0 -1: ax 1 -1 -> ax 2 +1

		d_same_sign = {(rotation_axis - rhr_direction) % 3: self.colors[(rotation_axis + rhr_direction) % 3]}
		new_colors.update(d_same_sign)

		d_to_switch = self.colors[(rotation_axis - rhr_direction) % 3]
		switched_d = {-1: d_to_switch[1], 1: d_to_switch[-1]}
		d_opposite_sign = {(rotation_axis + rhr_direction) % 3: switched_d}
		new_colors.update(d_opposite_sign)

		self.colors = new_colors

	def is_visible(self):
		return any(x in [0, self.n_on_side - 1] for x in self.coordinates)

	def visible_str(self):
		s = self.str()
		new = list(s[:3] + "_" * 3)
		for i in range(3):
			if int(s[i]) in [0, self.n_on_side - 1]:
				new[i + 3] = s[i + 3]
		# print(new)
		return "".join(new)

	def str(self):
		s = "{0}{1}{2}".format(*self.coordinates)
		s += "{0}{1}{2}".format(self.colors[0][1], self.colors[1][1], self.colors[2][1])
		return s

	def __eq__(self, other):
		if type(other) is CubePiece:
			return self.visible_str() == other.visible_str()
			# return self.str() == other.str()
		return NotImplemented

	def __lt__(self, other):
		return self.str() < other.str()


class Cube:
	def __init__(self, n_on_side):
		self.n_on_side = n_on_side
		self.pieces = {"{0}{1}{2}".format(i, j, k): CubePiece((i, j, k), n_on_side) for k in range(self.n_on_side) for j in range(self.n_on_side) for i in range(self.n_on_side)}

	def apply_move(self, rotation_axis, layer_index, rhr_direction):
		assert rotation_axis in [0, 1, 2]
		assert rhr_direction in [-1, 1]
		assert 0 <= layer_index < self.n_on_side

		# print("applying move")
		for piece in self.pieces.values():
			# s0 = piece.str()
			if piece.coordinates[rotation_axis] == layer_index:
				piece.rotate(rotation_axis, rhr_direction)
			# print("{0} -> {1}".format(s0, piece.str()))

		self.rearrange_pieces()

	def rearrange_pieces(self):
		new_pieces = {}
		# print("rearranging")
		for piece in self.pieces.values():
			# print(piece.str())
			new_coords = piece.str()[:3]
			new_pieces[new_coords] = piece
		self.pieces = new_pieces

	def get_random_move(self):
		return (random.choice(range(3)), random.choice(range(self.n_on_side)), random.choice([-1, 1]))

	def apply_random_move(self):
		move = self.get_random_move()
		self.apply_move(*move)

	def scramble(self):
		for i in range(1000):
			self.apply_random_move()

	def is_solved(self):
		orientation = [None, None, None]
		for piece in self.pieces.values():
			s = piece.visible_str()
			for i, x in enumerate(s[3:]):
				if x != "_":
					if orientation[i] is None:
						orientation[i] = x
					elif orientation[i] != x:
						return False
		return True

	def __eq__(self, other):
		if type(other) is Cube:
			if self.n_on_side != other.n_on_side:
				return False
			for coords in self.pieces:
				assert self.pieces[coords].str()[:3] == coords
				assert other.pieces[coords].str()[:3] == coords
				if self.pieces[coords] != other.pieces[coords]:
					# print("failed equality at piece {0}".format(self.pieces[coords].str()))
					return False
			return True

		return NotImplemented


def measure_moveset_period(cube, moveset):
	i = 0
	new_c = deepcopy(cube)
	while True:
		for move in moveset:
			new_c.apply_move(*move)
		i += 1
		if new_c == cube:
			return i
		elif i > 100000:
			raise RuntimeError("max period exceeded")

def measure_time_to_solve(cube):
	i = 0
	while not cube.is_solved():
		cube.apply_random_move()
		i += 1
		if i > 100000:
			raise RuntimeError("max moves exceeded in solving")
	return i


c = Cube(3)
# c.scramble()  # can actually affect period, e.g. rotation among the 4 centers of a face is indistinguishable if cube starts solved, but not in general

# moves = [(0, 0, 1)]
# moves = [(0, 0, 1), (1, 0, 1), (0, 3, -1), (1, 3, -1)]
# moves = [(2, 3, -1), (1, 0, -1)]

periods = []
for i in range(1000):
	moves = [c.get_random_move() for i in range(3)]
	period = measure_moveset_period(c, moves)
	# print("moveset {0}\nhas period {1}".format(moves, period))
	periods.append(period)

print(sorted(Counter(periods).items()))

# print(measure_time_to_solve(c))







