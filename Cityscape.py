import math
import random

from PIL import Image


class Building:
	def __init__(self, top_left_corner, r_range, c_range, height):
		self.top_left_corner = top_left_corner
		self.r_range = r_range
		self.c_range = c_range
		self.height = height
		self.outer_columns = self.get_outer_columns()
		self.rc_center = ((self.top_left_corner[0] + self.r_range * 1.0/2), (self.top_left_corner[1] + self.c_range * 1.0/2))
		self.center_column = Column(self.rc_center[0], self.rc_center[1], self.height)
		self.horizontal_edges = self.get_horizontal_edges()
		self.corners = self.get_corners()

	def get_outer_columns(self):
		result = []
		for r in [self.top_left_corner[0], self.top_left_corner[0] + self.r_range]:
			for c in [self.top_left_corner[1], self.top_left_corner[1] + self.c_range]:
				result.append(Column(r, c, self.height))
		return result

	def get_corners(self):
		return self.get_top_corners() + self.get_bottom_corners()

	def get_top_corners(self):
		return [column.top for column in self.outer_columns]

	def get_bottom_corners(self):
		return [column.bottom for column in self.outer_columns]

	def get_hidden_corners(self, observer):
		# WLOG x_size and y_size = 1 (we get floats from the rcz-xy conversion)
		z0 = observer[2]
		result = []
		if z0 <= self.height:
			top_corner = None
			max_top_corner_distance = -1
			for corner in self.get_top_corners():
				d = get_rcz_distances(corner, observer)[3]
				if d > max_top_corner_distance:
					top_corner = corner
					max_top_corner_distance = d
			result.append(top_corner)
		if z0 >= 0:
			bottom_corner = None
			max_bottom_corner_distance = -1
			for corner in self.get_bottom_corners():
				d = get_rcz_distances(corner, observer)[3]
				if d > max_bottom_corner_distance:
					bottom_corner = corner
					max_bottom_corner_distance = d
			result.append(bottom_corner)
		return result

	def get_neighboring_columns(self, reference_column):
		result = []
		r0, c0 = reference_column.r_base, reference_column.c_base
		for column in self.outer_columns:
			r, c = column.r_base, column.c_base
			if (r != r0 and c == c0) or (r == r0 and c != c0):
				result.append(column)
		return result

	def get_horizontal_edges(self):
		top_edges = []
		bottom_edges = []
		for column in self.outer_columns:
			neighbors = self.get_neighboring_columns(column)
			for neighbor in neighbors:
				top_edge = Edge(column.top, neighbor.top)
				if any([top_edge.is_equivalent_to_edge(e) for e in top_edges]):
					pass
				else:
				    top_edges.append(top_edge)

				bottom_edge = Edge(column.bottom, neighbor.bottom)
				if any([bottom_edge.is_equivalent_to_edge(e) for e in bottom_edges]):
					pass
				else:
				    bottom_edges.append(bottom_edge)
		return top_edges + bottom_edges

	def get_distance_to_observer(self, observer):
		return min([get_rcz_distances(corner, observer)[3] for corner in self.corners])


class Column:
	def __init__(self, r_base, c_base, height):
		self.r_base = r_base
		self.c_base = c_base
		self.height = height
		self.top = (self.r_base, self.c_base, self.height)
		self.bottom = (self.r_base, self.c_base, 0)
		self.edge_representation = Edge(self.top, self.bottom)

	def get_top_and_bottom_xy(self, observer, x_size, y_size):
		return self.edge_representation.get_end_xys(observer, x_size, y_size)


class Edge:
	def __init__(self, end1, end2):
		self.end1 = end1
		self.end2 = end2
		r1, c1, z1 = end1
		r2, c2, z2 = end2

	def get_end_xys(self, observer, x_size, y_size):
		xy1 = get_xy_from_rcz_pairs(self.end1, observer, x_size, y_size)
		xy2 = get_xy_from_rcz_pairs(self.end2, observer, x_size, y_size)
		return (xy1, xy2)

	def is_equivalent_to_edge(self, edge):
		return (self.end1 == edge.end1 and self.end2 == edge.end2) or (self.end1 == edge.end2 and self.end2 == edge.end1)


class City:
	def __init__(self, r_size, c_size):
		self.r_size = r_size
		self.c_size = c_size
		self.buildings = []
		self.array = [[0 for c in range(self.c_size)] for r in range(self.r_size)]

		self.generate_buildings()

	def generate_buildings(self):
		for r in range(self.r_size):
			for c in range(self.c_size):
				if random.random() < 5:
					building = Building((r+0.1, c+0.1), 0.8, 0.8, 5)
					self.buildings.append(building)
					building_index = len(self.buildings)
					self.array[r][c] = building_index

	def get_buildings_in_order_of_distance(self, observer, far_to_close=True):
		return sorted(self.buildings, key=lambda x: x.get_distance_to_observer(observer), reverse=far_to_close)

	def print_array(self):
		print("\n".join([repr(i) for i in self.array]))

	def get_image(self, observer, x_size, y_size):
		dims = (x_size, y_size)
		background_rgb = (255,255,255)
		im = Image.new("RGBA",dims,background_rgb)
		for building in self.get_buildings_in_order_of_distance(observer, far_to_close=True):
			color_rgb = tuple([random.randint(0,200) for i in range(3)])
			hidden_corners = building.get_hidden_corners(observer)
			for column in building.outer_columns:
				if column.top not in hidden_corners and column.bottom not in hidden_corners:
					top_xy, bottom_xy = column.get_top_and_bottom_xy(observer, x_size, y_size)
					x_range, y_range = get_xy_ranges_from_xy_ends(top_xy, bottom_xy, x_size, y_size)
					for x in x_range:
						for y in y_range:
							im.putpixel((x, y), color_rgb)
			for edge in building.horizontal_edges:
				end1, end2 = edge.end1, edge.end2
				if end1 not in hidden_corners and end2 not in hidden_corners:
					end1_xy, end2_xy = edge.get_end_xys(observer, x_size, y_size)
					path = get_xy_path_from_xy_ends(end1_xy, end2_xy, x_size, y_size)
					for xy in path:
						x, y = xy
						im.putpixel((x, y), color_rgb)
		return im

def get_xy_path_from_xy_ends(end1, end2, x_size, y_size):
	# can never have more pixels than its own length in pixels, so be safe and get a pixel for n*2 pixel-lengths
	xy_length = math.sqrt((end1[0]-end2[0])**2 + (end1[1]-end2[1])**2)
	raw_xy_points = []
	for i in range(math.ceil(xy_length) + 1):
		alpha = i * 1.0/xy_length
		x = end1[0] + alpha * (end2[0] - end1[0])
		y = end1[1] + alpha * (end2[1] - end1[1])
		raw_xy_points.append((x,y))
	result = []
	for p in raw_xy_points:
		x = int(round(p[0],0))
		y = int(round(p[1],0))
		if x < 0 or x > x_size or y < 0 or y > y_size:
			pass
		else:
		    result.append((x,y))
	return result

def get_xy_ranges_from_xy_ends(end1, end2, x_size, y_size):
	max_x = int(round(max(end1[0], end2[0]), 0))
	min_x = int(round(min(end1[0], end2[0]), 0))
	x_range = [i for i in range(min_x, max_x + 1) if i >= 0 and i <= x_size-1]

	max_y = int(round(max(end1[1], end2[1]), 0))
	min_y = int(round(min(end1[1], end2[1]), 0))
	y_range = [i for i in range(min_y, max_y + 1) if i >= 0 and i <= y_size-1]

	return (x_range, y_range)

def get_rcz_distances(p1, p2, first_minus_second=True):
	r1, c1, z1 = p1
	r2, c2, z2 = p2
	dr = r1 - r2 if first_minus_second else r2 - r1
	dc = c1 - c2 if first_minus_second else c2 - c1
	dz = z1 - z2 if first_minus_second else z2 - z1
	d = math.sqrt(dr**2 + dc**2 + dz**2)
	return dr, dc, dz, d

def get_xy_from_rcz_pairs(target, observer, x_size, y_size):
	dr, dc, dz, d = get_rcz_distances(target, observer, first_minus_second=True)

	if dr >= 0:
		# x_ = 0
		raise ValueError("Observer must be in a higher_numbered row from target")
	else:
	    x_ = -1 * math.atan(dc * 1.0/dr)
	if d == 0:
		# y_ = 0
		raise ValueError("Observer must be in a different place from target")
	else:
	    y_ = math.asin(dz * 1.0/d)

	# define x = 0 as 90 deg left, x_size as 90 deg right
	# define y = 0 as 90 deg up,   y_size as 90 deg down
	x = (x_ + math.pi/2) * (x_size * 1.0/math.pi)
	y = (y_ - math.pi/2) * (y_size * -1.0/math.pi)

	return (x, y)


C = City(40,40)
# C.print_array()
im = C.get_image((41,20,2.5), 500, 500)
im.save("Cityscape.png")