import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
        self.demand *= np.exp(random.random() - 0.5)
        self.demand = max(0, self.demand)

    def __repr__(self):
        return "I({}, {})".format(self.x, self.y)


class IntersectionGrid:
    def __init__(self, intersections):
        self.intersections = intersections
        self.x_max = max(p.x for p in intersections)
        self.y_max = max(p.y for p in intersections)
        self.grid = self.get_grid_from_intersections(intersections)

    def get_grid_from_intersections(self, intersections):
        grid = np.full(shape=(self.x_max + 1, self.y_max + 1), fill_value=None)  # why are these args backwards
        for p in intersections:
            grid[p.x, p.y] = p
        return grid

    def get_intersection_at_coordinates(self, x, y):
        return self.grid[x, y]

    def contains(self, coords):
        return 0 <= coords[0] <= self.x_max and 0 <= coords[1] <= self.y_max


class RoadSegment:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.capacity = int(euclidean_distance(self.a, self.b) * 5)
        self.n_cars = random.randint(1, self.capacity)  # no empty streets at first, so there will always be at least 1 car in existence

    def other_end_from_point(self, p):
        if p == self.a:
            return self.b
        elif p == self.b:
            return self.a
        else:
            raise ValueError("point not on this road segment")

    def is_full(self):
        assert 0 <= self.n_cars <= self.capacity
        return self.n_cars == self.capacity

    def is_empty(self):
        assert 0 <= self.n_cars <= self.capacity
        return self.n_cars == 0

    def fullness(self):
        return self.n_cars / self.capacity

    def __eq__(self, other):
        if type(other) is not RoadSegment:
            raise TypeError("comparing RoadSegment to object of other type: {}".format(type(other)))
        ours = (self.a, self.b)
        theirs1 = (other.a, other.b)
        theirs2 = (other.b, other.a)
        return ours == theirs1 or ours == theirs2

    def __hash__(self):  # Python 3 requires this if you define __eq__
        return hash((self.a, self.b))


class RoadNetwork:
    def __init__(self, intersection_grid):
        self.intersection_grid = intersection_grid
        self.segments = self.get_segments()

    def get_segments(self):
        segments = set()
        points = set(self.intersection_grid.intersections)
        point_coordinates = [(p.x, p.y) for p in points]
        points_reached = set()
        points_left = points - points_reached  # setminus
        start_point = random.choice(list(points_left))
        points_reached.add(start_point)
        a = start_point
        while len(points_left) > 0:
            # random walk until find another point in the grid, even if it is already in the network (not trying to create an acyclic graph)
            p = (a.x, a.y)
            while p not in point_coordinates or p == (a.x, a.y):
                d = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                new_p = (p[0] + d[0], p[1] + d[1])
                if self.intersection_grid.contains(new_p):
                    p = new_p
                    # else continue
            new_point = self.intersection_grid.get_intersection_at_coordinates(*p)
            segments.add(RoadSegment(a, new_point))
            points_reached.add(new_point)
            points_left = points - points_reached  # setminus
            a = new_point # start from this point next time, to speed up search so we are not just retracing old roads
            # be careful not to mutate any of these objects that are references to intersection_grid's intersections or such

        return segments

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

        # best_option = min(options, key=get_distance_if_taking_option)
        best_option = random.choice(options)  # try this first
        return best_option

    def get_segments_adjacent_to_point(self, p):
        return [x for x in self.segments if x.a == p or x.b == p]


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
                if random.random() < 0.9:
                    intersections.append(Intersection(x, y))
        return intersections

    def update(self):
        for p in self.intersections:
            p.update()
        max_demand = max(p.demand for p in self.intersections)

        # normalize demand to [0, 1]
        for p in self.intersections:
            p.demand /= max_demand

    def plot(self):
        colormap = cm.YlOrRd
        for segment in self.road_network.segments:
            plt.plot([segment.a.x, segment.b.x], [segment.a.y, segment.b.y], color=colormap(segment.fullness()))
        for p in self.intersections:
            plt.scatter([p.x], [p.y], c=colormap(p.demand))


class Agent:
    def __init__(self, start_segment, destination):
        self.location = random.choice([start_segment.a, start_segment.b])
        self.start_segment = start_segment
        self.destination = destination
        self.is_parked = False
        self.start_segment.n_cars -= 1
        self.blocks_since_destination = None

    @staticmethod
    def pick_start_segment(road_network):
        assert len(road_network.segments) > 0
        candidates = [x for x in road_network.segments if not x.is_empty()]
        return random.choice(candidates)

    @staticmethod
    def pick_destination(intersections):
        total_demand = sum(x.demand for x in intersections)
        r = random.uniform(0, total_demand)
        g = (x for x in intersections)
        candidate = next(g)
        while r >= candidate.demand:
            r -= candidate.demand
            candidate = next(g)
        return candidate

    def move_toward_destination(self, road_network):
        segment = road_network.get_most_direct_segment(self.location, self.destination)
        self.location = segment.other_end_from_point(self.location)
        if self.location == self.destination:
            self.blocks_since_destination = 0

    def look_for_parking_nearby(self, road_network):
        options = road_network.get_segments_adjacent_to_point(self.location)
        choice = random.choice(options)
        # have to go down the street to check for parking
        self.location = choice.other_end_from_point(self.location)
        self.blocks_since_destination += 1
        if not choice.is_full():
            self.park(choice)

    def seek_parking(self, road_network):
        if self.blocks_since_destination is None:
            # destination has not been reached yet
            self.move_toward_destination(road_network)
        else:
            self.look_for_parking_nearby(road_network)

    def park(self, choice):
        choice.n_cars += 1
        self.is_parked = True
        

def euclidean_distance(p1, p2):
    p1a = np.array([p1.x, p1.y])
    p2a = np.array([p2.x, p2.y])
    d = p2a - p1a
    return np.sqrt(np.sum(d ** 2))



if __name__ == "__main__":
    city = City(10, 10)
    city.update()

    plt.ion()
    city.plot()

    for i in range(1000):
        agent = Agent(Agent.pick_start_segment(city.road_network), Agent.pick_destination(city.intersections))
        while not agent.is_parked:
            agent.seek_parking(city.road_network)
        city.update()

        if i % 50 == 0:  # redrawing everything too often is the greatest performance hit
            plt.gcf().clear()
            city.plot()
            plt.pause(0.01)