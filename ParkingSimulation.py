import random
import numpy as np
import matplotlib.pyplot as plt


# have a grid of points, joined randomly by streets (somehow ensure whole graph is connected)
# each point has an amount of demand, representing the probability that someone will try to go there
# at random times, agents will set out from a location of a car and go to a destination (don't keep track of them before/after the trip)
# agents will drive to their destination (need to create pathfinding algorithm) then find a place to park nearby.
# each segment of road has a fixed number of parking spots and tracks how many cars are there.
# plot destinations as circles with color corresponding to demand
# plot segments of roads with color corresponding to crowdedness
# possibly plot current location of agents


class Intersection:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.demand = random.random()

    def update(self):
        self.demand += random.random()


class IntersectionGrid:
    def __init__(self, intersections):
        self.intersections = intersections
        self.grid = IntersectionGrid.get_grid_from_intersections(intersections)

    @staticmethod
    def get_grid_from_intersections(intersections):
        x_max = max(p.x for p in intersections)
        y_max = max(p.y for p in intersections)
        grid = np.full(shape=(x_max + 1, y_max + 1), fill_value=None)  # why are these args backwards
        for p in intersections:
            grid[p.x, p.y] = p
        return grid

    def get_intersection_at_coordinates(self, x, y):
        return self.grid[x, y]


class RoadSegment:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def other_end_from_point(self, p):
        if p == self.a:
            return self.b
        elif p == self.b:
            return self.a
        else:
            raise ValueError("point not on this road segment")

    def __eq__(self, other):
        ours = (self.a, self.b)
        theirs1 = (other.a, other.b)
        theirs2 = (other.b, other.a)
        return ours == theirs1 or ours == theirs2

    def __hash__(self):  # Python 3 requires this if you define __eq__
        return hash((self.a, self.b))


class RoadNetwork:
    def __init__(self, intersection_grid):
        self.intersection_grid = intersection_grid

        segments = set()
        points = set(intersection_grid.intersections)
        point_coordinates = [(p.x, p.y) for p in points]
        points_reached = set()
        points_left = points - points_reached  # setminus
        start_point = random.choice(list(points_left))
        points_reached.add(start_point)
        while len(points_left) > 0:
            a = random.choice(list(points_left))
            # random walk until find another point in the grid, even if it is already in the network (not trying to create an acyclic graph)
            p = (a.x, a.y)
            while p not in point_coordinates or p == (a.x, a.y):
                d = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                p = (p[0] + d[0], p[1] + d[1])
            new_point = intersection_grid.get_intersection_at_coordinates(*p)
            segments.add(RoadSegment(a, new_point))
            points_reached.add(new_point)
            points_left = points - points_reached  # setminus
            # be careful not to mutate any of these objects that are references to intersection_grid's intersections or such

        self.segments = segments



    def get_road_segment_from_point_toward_destination(origin, destination):
        if random.random() < 0.75:
            return self.get_most_direct_segment(origin, destination)
        else:
            return random.choice(segments_from_point)

    def get_most_direct_segment_by_bearing(self, origin, destination):
        raise Exception("do not use")
        dx = destination.x - origin.x
        dy = destination.y - origin.y
        assert dx != 0 or dy != 0
        if abs(dx) == abs(dy):
            dx += random.choice(-1, 1) * 1e-6  # hack to make it choose one of the two equally close directions
        if abs(dx) > abs(dy):
            assert dx != 0
            if dx > 0:
                closest_segment_to_bearing = "east"
            else:
                closest_segment_to_bearing = "west"
        else:
            assert dy != 0
            if dy > 0:
                closest_segment_to_bearing = "north"
            else:
                closest_segment_to_bearing = "south"
        return closest_segment_to_bearing  # should change this to return an actual segment object if I decide to use this method

    def get_most_direct_segment(self, origin, destination):
        options = self.get_segments_adjacent_to_point(origin)

        def get_distance_if_taking_option(option):
            return euclidean_distance(option.other_end_from_point(origin), destination)

        best_option = min(options, key=get_distance_if_taking_option)
        return best_option


class City:
    def __init__(self, x_max, y_max):
        self.intersections = City.get_intersections(x_max, y_max)
        self.intersection_grid = IntersectionGrid(self.intersections)
        self.road_network = RoadNetwork(self.intersection_grid)

    @staticmethod
    def get_intersections(x_max, y_max):
        intersections = []
        for x in range(x_max):
            for y in range(y_max):
                if random.random() < 1:#0.3:
                    intersections.append(Intersection(x, y))
        return intersections

    def plot(self):
        for p in self.intersections:
            plt.scatter([p.x], [p.y]) # color=p.demand)  # somehow define a cmap later, but need bounds (could normalize all demands on each step)


class Agent:
    def __init__(self):
        pass


def euclidean_distance(p1, p2):
    p1a = np.array(p1.x, p1.y)
    p2a = np.array(p2.x, p2.y)
    d = p2a - p1a
    return np.sqrt(np.sum(d ** 2))



if __name__ == "__main__":
    city = City(1, 2)

    city.plot()
    plt.show()