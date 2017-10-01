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
        self.demand = random.random()

    def update(self):
        self.demand += random.random()


class IntersectionGrid:
    def __init__(self, intersections):
        self.grid = IntersectionGrid.get_grid_from_intersections(intersections)

    @staticmethod
    def get_grid_from_intersections(intersections):
        x_max = max(p.x for p in intersections)
        y_max = max(p.y for p in intersections)
        grid = np.full(shape=(x_max, y_max), fill_value=None)  # why are these args backwards
        for p in intersections:
            grid[p.x, p.y] = p
        return grid

    def get_intersection_at_coordinates(x, y):
        return self.grid[x, y]


class RoadSegment:
    def __init__(self, p1, p2):
        pass


class RoadNetwork:
    def __init__(self, intersection_grid):
        self.intersection_grid = intersection_grid

    def get_road_segment_from_point_toward_destination(origin, destination):
        if np.random.random() < 0.75:
            return self.get_most_direct_segment(origin, destination)
        else:
            return np.random.choice(segments_from_point)

    def get_most_direct_segment_by_bearing(self, origin, destination):
        raise Exception("do not use")
        dx = destination.x - origin.x
        dy = destination.y - origin.y
        assert dx != 0 or dy != 0
        if abs(dx) == abs(dy):
            dx += np.random.choice(-1, 1) * 1e-6  # hack to make it choose one of the two equally close directions
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
        best_option = min(x for x in options, key=lambda x: euclidean_distance(x.other_end_from_point(origin), destination))


class City:
    def __init__(self, x_max, y_max):
        self.intersections = City.intersections(x_max, y_max)
        self.intersection_grid = IntersectionGrid(self.intersections)
        self.road_network = RoadNetwork(self.intersection_grid)

    @staticmethod
    def get_intersections(x_max, y_max):
        intersections = []
        for x in range(x_max):
            for y in range(y_max):
                if np.random.random() < 0.3:
                    intersections.append(Intersection(x, y))
        return intersections

    def plot(self):
        for intersection in self.intersections:


class Agent:
    def __init__(self):
        pass


def euclidean_distance(p1, p2):
    p1a = np.array(p1.x, p1.y)
    p2a = np.array(p2.x, p2.y)
    d = p2a - p1a
    return np.sqrt(np.sum(d ** 2))



if __name__ == "__main__":
    city = City(10, 10)

    city.plot()