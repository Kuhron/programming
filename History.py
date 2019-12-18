# NOTE TO SELF ON COORDINATES: iterating through x and y is fine, but ideally everything should use (row, col) coords!
# DO NOT REFER TO POINTS DIRECTLY WITH (x,y); use (r,c).

# R-size is number of rows
# C-size is number of columns
# intuitive enough, but I have confused myself with this many times in the past

# use "rad" for radius

import argparse
import math
import random
import sys
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from mpl_toolkits.mplot3d.axes3d import *
from matplotlib import cm


def stop():
    sys.exit()

def prompt(s):
    return input(s+" (y/n; default = no) ") == "y"

def grid_minimum(lst_of_lsts):
    return min([min(lst) for lst in lst_of_lsts])

def d(p1,p2): 
   return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def all_equal(iterator):
    # source: http://stackoverflow.com/questions/3844801/
    return len(set(iterator)) <= 1

def lengths_match(lst_of_lsts):
    return all_equal([len(i) for i in lst_of_lsts])

def unique(lst):
    result = []
    for i in lst:
        if i not in result:
            result.append(i)
    return result

def population_dist(birth_or_death="birth"):
    if birth_or_death == "birth":
        #return 3*(9+np.random.zipf(1.5)) # min 30 max +Inf
        return random.choice(range(30,101))
    elif birth_or_death == "death":
        return random.choice(range(20,101))
    else:
        print("function population_dist: arg birth_or_death must be either \"birth\" or \"death\".")

def grid_circle(r_size,c_size,r_0,c_0,rad):
    """
    Takes the grid's size, the coordinates of the center point, and the radius.
    Returns the set of points in the (integer-only) grid within the radius of that point.
    """

    # restrict to the square inscribed by the grid circle in order to make algorithm faster
    # make ends of range ceil functions to capture decimals; e.g. range(0.3,2.5) should yield [1,2], which is range(1,3)
    # except on right end, when arg is int,                  e.g. range(0.3,3)   should yield [1,2], have same thing
    possible_rows = [i for i in range(math.ceil(max(0,r_0-rad)),math.floor(min(r_size-1,r_0+rad))+1)]
    possible_cols = [i for i in range(math.ceil(max(0,c_0-rad)),math.floor(min(c_size-1,c_0+rad))+1)]
    
    result = [] # keep blank lists as elements for rows with no matching elements
    for row in range(r_size):
        
        if row not in possible_rows:
            # faster than calculating just to find that the left edge is right of the right edge (so the range is empty)
            result.append([])
            continue
        if row == r_0:
            result.append([i for i in possible_cols])
            continue
        
        # would rather leave this case to the edge-finding method below in case of non-integer radii
        r_separation = abs(row - r_0)

        # imaginary bug prevention
        if r_separation > rad:
            print("function grid_circle: an impossible row ({0}) is being considered for column selection".format(row))
            print("possible rows: {0}, with r_0 = {1} and rad = {2}".format(possible_rows,r_0,rad))
            stop()
        
        # if row == r_0 + rad or row == r_0 - rad:
        #     result.append([c_0])
        #     continue

        # this method is slow because it is using the distance function unnecessarily many times
        # result.append([c for c in filter(
        #     lambda c: d((r_0,c_0),(row,c)) <= rad, [i for i in possible_cols]
        # )])

        # improved method by finding the column edges corresponding to each row and then just taking range between them
        # solve distance function rather than calculating the distance itself
        
        # k = "int"
        # if row > r_0:
        #     k = 1
        # elif row < r_0:
        #     k = -1
        edge_separation_from_center = math.floor(math.sqrt(rad**2 - r_separation**2))
        
        left_edge = c_0 - edge_separation_from_center
        left_edge = max(0,left_edge)
        right_edge = c_0 + edge_separation_from_center
        right_edge = min(c_size-1,right_edge)

        result.append([i for i in range(left_edge,right_edge+1)])

    return result

def grid_line(r_size, c_size, p1, p2):
    """
    Takes two points, returns list of row lists containing the column indices on the line between them.
    (catty corner only, no extra connections).
    E.g. [[2], [3], [4]]
    """

    # coordinates are now returned as tuples, so change to list in case of needing to flip
    p1 = list(p1)
    p2 = list(p2)

    # use point-slope form to get equation of line through both points
    # then use row number to get column number on the line 
    r_range = p2[0]-p1[0]
    original_r_range = r_range
    #print("r_range",r_range)
    c_range = p2[1]-p1[1]
    original_c_range = c_range
    #print("c_range",c_range)

    flip_it = abs(r_range) < abs(c_range)

    # because result was assigned differently in the if blocks, Python was complaining about referencing local variables before assignment
    result = []

    # special case of both on the same row, to avoid division by zero
    if r_range == 0:
        result = [[] for r in range(r_size)]
        result[p1[0]] = [i for i in range(min(p1[1],p2[1]),max(p1[1],p2[1])+1)]

    # if the slope is less than one, we must transpose everything to make the set, then transpose it back
    if flip_it:
        holder = r_size
        r_size = c_size
        c_size = holder

        holder = r_range
        r_range = c_range
        c_range = holder
        #print("r_range",r_range)
        #print("c_range",c_range)

        holder = p1[0]
        p1[0] = p1[1]
        p1[1] = holder
        #print("p1",p1[0],p1[1])

        holder = p2[0]
        p2[0] = p2[1]
        p2[1] = holder
        #print("p2",p2[0],p2[1])



    if r_range != 0:
        m = float(c_range)/r_range
        # y-y = m*(x-x)
        # ==> c = m*(r-r_ref)+c_ref

        result = []
        for row in range(r_size):
            #print("row", row)
            if row > max(p1[0],p2[0]) or row < min(p1[0],p2[0]):
                # if the row is not between the points, add an empty list
                #print("skipped")
                result.append([])
                continue
            #a = [c for c in filter(lambda c: False, [i for i in range(c_size)])]
            #print("actual value {0}".format(m*(row-p1[0])+p1[1]))
            #print("rounded to   {0}".format(round(m*(row-p1[0])+p1[1])))
            result.append([round(m*(row-p1[0])+p1[1])])

    # switch r_range and c_range back because this was causing the result-converting block not to run
    if flip_it:
        holder = r_range
        r_range = c_range
        c_range = holder

    #print(result)

    # transposing back and also converting row and column values appropriately (see below)
    if flip_it:
        # this converts, e.g., [[0],[0],[1],[1],[]] to [[0,1],[2,3],[],[],[]]
        # the return value should list in its rows i, which rows j in the current result have value i
        
        flipped_result = [[] for c in range(c_size)]
        for r in range(len(result)):
            if result[r] != []:
                flipped_result[result[r][0]].append(r)
        return flipped_result

    return result

def neighbors(r_size, c_size, r_0, c_0, n=8):
    """
    Finds the 4 (cardinal directions) or 8 (cardinal and secondary directions) neighbors of the point (r_0, c_0).
    Returns a row-major list of lists including column indices for each row list.
    """
    if n not in [4,8]:
        print("function neighbors: choose n in [4,8]")
        return
   
    if n == 4:
        rad = 1
    elif n == 8:
        # just use 1.5 to get the sqrt(2) ones but not the 2 ones
        rad = 1.5

    # get_circle automatically ignores things off the edge
    result = grid_circle(r_size, c_size, r_0, c_0, rad)
    # remove the reference point itself
    try:
        result[r_0].remove(c_0)
    except ValueError:
        print("function neighbors: ValueError was raised, likely because {0} is not in the row {1}.".format(c_0, repr(result[r_0])))
        print("Values passed creating this error:\nr_size: {0}, c_size: {1}, r_0: {2}, c_0: {3}, n: {4}.".format(r_size,c_size,r_0,c_0,n))
        stop()
    return result

def is_on_shore(elevation_grid, r_0, c_0):
    """
    Checks if the point (r_0, c_0) is on the shore, defined as bordering a point with negative elevation.
    """
    if not lengths_match(elevation_grid):
        print("function is_on_shore: elevation grid row lengths do not match")
        return

    el = elevation_grid[r_0][c_0] # elevation of the point in question
    if el < 0:
        # if it's in the water, don't build cities
        return False

    r_size = len(elevation_grid)
    c_size = len(elevation_grid[0])

    neighbors_lst = neighbors(r_size, c_size, r_0, c_0, n=8)
    for row in range(r_size):
        for col in neighbors_lst[row]: # just look at the indices listed in each row's list
            if elevation_grid[row][col] < 0:
                return True
    return False

def is_critical_point_helper(elevation_grid, r_0, c_0, min_or_max):
    """
    Checks if the point (r_0, c_0) is at a lower or higher elevation than all of its neighbors.
    Returns either "minimum", "maximum", or "neither", for passing to is_local_minimum and is_local_maximum.
    Restricted to non-negative elevations to prevent underwater cities.
    """
    if not lengths_match(elevation_grid):
        print("function is_critical_point_helper: elevation grid row lengths do not match")
        return

    if min_or_max not in ["min","max"]:
        print("function is_critical_point_helper: please specify the fourth argument as \"min\" or \"max\".")

    r_size = len(elevation_grid)
    c_size = len(elevation_grid[0])

    el = elevation_grid[r_0][c_0] # elevation of the point in question, to be checked against the neighbors
    if el < 0:
        return False
    
    # the function as is calculates minima, so I'm just gonna multiply the inequality by -1 for maxima
    k = 1 if min_or_max == "min" else -1

    neighbors_lst = neighbors(r_size, c_size, r_0, c_0, n=8)
    for row in range(r_size):
        for col in neighbors_lst[row]: # just look at the indices listed in each row's list
            #print(elevation_grid[row][col])
            if k*elevation_grid[row][col] < k*el:
                # if any of the neighbors is lower, return False
                return False
    return True

def is_local_minimum(elevation_grid, r_0, c_0):
    return is_critical_point_helper(elevation_grid,r_0,c_0,"min")

def is_local_maximum(elevation_grid, r_0, c_0):
    return is_critical_point_helper(elevation_grid,r_0,c_0,"max")

def generate_name_global():
    """
    Creates super-fake fantasy-sounding word for city and kingdom names.
    """
    name = ""
    n_syll = random.choice(range(2,4))
    for i in range(n_syll):
        struct = ["X","V","X"]
        for i in [0,2]:
            if random.random() < 0.5:
                struct[i] = "C"

        for s in struct:
            if s == "C":
                name += random.choice(["m","n","p","b","t","d","k","g","f","v","s","z","sh","zh","x","gh","h","l","r","y","w","ch","j"])
            elif s == "V":
                name += random.choice(["a","e","i","o","u"])
    
    return name[0].upper() + name[1:]

def weighted_choice(d):
    # weighted choice algorithm (shouldn't some library contain this? or do I have to keep rewriting it)
    items = d.items()
    keys = [i[0] for i in items]
    values = [i[1] for i in items]
    #d_lst = sorted(d.keys())
    d_lst = keys
    total = sum(values)

    if total <= 0: 
        # if no ability to weight choices because everyone just had 0 of every resource; less than should not happen, but still
        # cannot do anything, so don't move anyone
        return

    #print(total)
    r = random.random()*total
    #print(r)
    i = 0
    result = None
    while True:
        e = d[d_lst[i]]
        if r < e:
            result = d_lst[i]
            break
        else:
            r -= e
            i += 1

    return result

# def hash_to_unit_interval(s): # certainly doesn't work yet
#     n = hash(s)
#     u = float(n)/(10**len(str(n)))
#     #return u/2.0 + 0.5
#     #return u % 1
#     return random.choice([-1,1])

def poisson_probability_of_n_events_in_time_dt(n, dt, lam):
    return 1.0/math.factorial(n) * (lam * dt)**n * math.exp(-1 * lam * dt)

class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.r, self.c = self.coordinates
        self.resources = {}
        self.stockpiles = {}

    def add_resource_amount(self, resource_name, amount):
        if resource_name not in self.resources:
            self.resources[resource_name] = 0
        self.resources[resource_name] += amount

    def add_stockpile_amount(self, resource_name, amount):
        if resource_name not in self.stockpiles:
            self.stockpiles[resource_name] = 0
        self.stockpiles[resource_name] += amount

    def get_resource_score(self):
        result = 0
        for resource in set(self.resources.keys()) - {"population"}: # do not include population in this at all
            result += (1+self.resources[resource])**0.5 - 1
        return result

    def get_stockpile_score(self):
        result = 0
        for resource in set(self.stockpiles.keys()) - {"population"}:
            result += (1+self.stockpiles[resource])**0.5 - 1 # require f(0) = 0
        result += (1 + self.stockpiles["population"])*100 # make people concentrate in populous cities (for testing this function)
        return result

    def get_battle_score(self):
        # make this more detailed later
        return self.stockpiles["guns"]

    def __getitem__(self, i):
        if i == 0:
            return self.r
        elif i == 1:
            return self.c
        else:
            raise IndexError("Point can only be indexed by 0 (row) or 1 (column).")


class City:
    def __init__(self,city_dct):
        self.map = city_dct["map"]
        self.coordinates = (city_dct["r"], city_dct["c"])
        self.point = self.map.points[self.coordinates]
        self.r = self.point.r
        self.c = self.point.c
        self.x = self.c
        self.y = self.map.r_size-self.r-1
        self.resources = self.point.resources
        self.stockpiles = self.point.stockpiles
        
        self.point.add_stockpile_amount("population", city_dct["population"])
        self.name = city_dct["name"]
        self.state = None
        self.trade_neighbors = None

    def contained_in_state(self,state):
        return self.state == state

    def get_routes(self):
        return [route for route in self.map.routes[self].values() if route is not None]

    def get_trade_routes(self):
        return [route for route in self.map.trade_routes if self in route.cities and route is not None]

    def get_land_trade_routes(self):
        return [route for route in self.map.land_trade_routes if self in route.cities and route is not None]

    def get_navigable_routes(self):
        return [route for route in self.get_routes() if route.is_navigable(self.get_navigation_technologies())]

    def get_navigation_technologies(self):
        return [res for res in self.point.resources.keys() if self.point.stockpiles[res] > 0]

    def get_trade_neighbors(self):
        return self.trade_neighbors if self.trade_neighbors else self.construct_trade_neighbors()

    def construct_trade_neighbors(self):
        result = []
        for pair in filter(lambda p: p.contains(self), self.state.land_trade_routes):
            partner_city = pair[0] if pair[1] is self else pair[1]
            result.append(partner_city)
        self.trade_neighbors = result
        return result

    def get_resource_score(self):
        return self.point.get_resource_score()

    def get_stockpile_score(self):
        return self.point.get_stockpile_score()

    def get_battle_score(self):
        return self.point.get_battle_score()

    def attempt_to_conquer(self, city_to_conquer):
        if self.state is city_to_conquer.state:
            # print("can't conquer city in same state")
            return

        s1 = self.get_battle_score()
        s2 = city_to_conquer.get_battle_score()
        diff = s1 - s2
        success = random.normalvariate(0, 1) < diff
        if success:
            self.conquer(city_to_conquer)

    def conquer(self, conquered_city):
        print("called conquer({0}, {1})".format(self.name, conquered_city.name))
        self.map.output("{0} from {1} conquered {2} from {3}.".format(
            self.name, self.state.name, conquered_city.name, conquered_city.state.name))
        # removal from old state
        conquered_city.state.cities.remove(conquered_city)
        conquered_city.state.refresh_routes()

        # addition to new state
        conquered_city.state = self.state
        self.state.cities.append(conquered_city)
        self.state.refresh_routes()

    def destroy(self):
        # beware that if any references remain then there may be a memory leak
        raise Exception("try replacing deletion of city instances with just an instance variable such as " \
            "self.is_active = False")
        self.map.cities.remove(self)
        self.map.types[self.point.r][self.point.c].remove("city")
        if "large_city" in self.map.types[self.point.r][self.point.c]:
            self.map.types[self.point.r][self.point.c].remove("large_city")
        if "largest_city" in self.map.types[self.point.r][self.point.c]:
            self.map.types[self.point.r][self.point.c].remove("largest_city")
        for route in self.get_routes():
            if route in self.map.trade_routes:
                self.map.trade_routes.remove(route)
            if route in self.map.land_trade_routes:
                self.map.land_trade_routes.remove(route)
        self.state.cities.remove(self)
        self.state.refresh_routes()
        self.state = None
        self.map = None


class State:
    def __init__(self,state_dct):
        self.map = state_dct["map"]
        self.cities = []
        self.name = state_dct["name"]
        self.display_name = state_dct["display_name"]
        self.network = state_dct["network"]
        self.population = state_dct["population"]
        self.geographic_center = state_dct["geographic_center"]
        self.designation = state_dct["designation"]
        self.routes = self.get_routes()
        self.trade_routes = self.get_trade_routes()
        self.land_trade_routes = self.get_land_trade_routes()
        self.propensity_to_conquer = random.random()

    def contains_city(self, city):
        return city.state is self

    def get_routes(self):
        return list(set([route for c in self.cities for route in c.get_routes()]))

    def get_trade_routes(self):
        return [route for route in self.map.trade_routes if all([c in self.cities for c in route.cities])]

    def get_land_trade_routes(self):
        return [route for route in self.map.land_trade_routes if all([c in self.cities for c in route.cities])]

    def get_all_navigable_routes(self):
        all_routes = []
        for city in self.cities:
            all_routes.extend(city.get_navigable_routes())
        return unique(all_routes)

    def get_internal_navigable_routes(self):
        return [route for route in self.get_all_navigable_routes() if self.contains_route(route)]

    def refresh_routes(self):
        self.routes = self.get_routes()
        self.trade_routes = self.get_trade_routes()
        self.land_trade_routes = self.get_land_trade_routes()

    def contains_route(self, route):
        return route.cities[0] in self.cities and route.cities[1] in self.cities

    def go_on_conquest(self):
        for city in self.cities:
            if random.random() < self.propensity_to_conquer:
                for route in city.get_navigable_routes():
                    city.attempt_to_conquer(route.get_other_city(city))


class Route: 
    def __init__(self, cities, r_size, c_size, types):
        self.cities = cities
        self.grid_line = grid_line(r_size, c_size, cities[0].coordinates, cities[1].coordinates)
        self.grid_line_types = [types[r][c] for r in range(r_size) for c in self.grid_line[r]]
        self.length = len(self.grid_line_types)

    def contains_type(self, typ):
        return typ in self.grid_line_types

    def get_grid_line_navigability(self, tech):
        result = []
        # d_true = {
        #     "default": {"land"}, # is a superset of shore, city, etc.
        #     "ships": {"water"},
        #     "climbing_gear": {"mountain"} # superset of volcano, peak
        # }
        d_false = {
            "default": {"water", "mountain"},
            "ships": {"land"},
            "climbing_gear": {"water"}
        }
        if tech not in d_false:
            return [False]*len(self.grid_line_types)
        for types_list in self.grid_line_types:
            # acceptable_types = d_true[tech]
            unacceptable_types = d_false[tech]
            # result.append(set(acceptable_types) & set(types_list) != set())
            result.append(set(unacceptable_types) & set(types_list) == set())

        return result

    def is_navigable(self, techs):
        techs = set(techs) | {"default"}
        navigabilities = {tech: self.get_grid_line_navigability(tech) for tech in techs}
        result = [any([navigabilities[tech][i] for tech in techs]) for i in range(self.length)]
        return all(result)

    def get_other_city(self, city):
        if city not in self.cities:
            raise IndexError("City {0} not part of route between {1} and {2}.".format(
                city.name, self.cities[0].name, self.cities[1].name))

        return self.cities[0] if city is self.cities[1] else self.cities[1]

    def contains(self, city):
        return city in self.cities

    def destroy(self):
        del self.cities[0].map.routes[self.cities[0]][self.cities[1]]
        del self.cities[0].map.routes[self.cities[1]][self.cities[0]]
        if self in self.cities[0].map.trade_routes:
            self.cities[0].map.trade_routes.remove(self)
        if self in self.cities[0].map.land_trade_routes:
            self.cities[0].map.land_trade_routes.remove(self)
        del self.cities

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.cities[i]


class Volcano:
    def __init__(self, point, name):
        self.point = point
        self.name = name
        self.average_power = random.paretovariate(0.5)
        # self.average_period = random.paretovariate(0.5) # power and frequency are independent
        self.average_period = self.average_power # power and frequency are inversely related (better for a pressure-inspired model)

    def will_erupt_this_period(self):
        lam = 1.0/ self.average_period
        p_no_eruption = poisson_probability_of_n_events_in_time_dt(0, 1, lam)
        return random.random() < 1 - p_no_eruption

    def erupt(self):
        if self.will_erupt_this_period():
            power = max(0, self.average_power * random.normalvariate(1, 0.5))
        else:
            power = 0
        return power


class Map:
    def __init__(self, r_size, c_size, output_mode):
        self.r_size = r_size
        self.c_size = c_size
        self.output_mode = output_mode
        self.output_string = ""
        self.points = {(r,c): Point((r,c)) for r in range(self.r_size) for c in range(self.c_size)}
        self.types = [[[] for i in range(c_size)] for r in range(r_size)]
        self.cities = []
        self.volcanoes = []
        self.largest_city = [-1,-1]
        self.max_population = 0
        self.used_names = [] # list of names generated from generate_name() so we don't repeat them
        self.names = self.generate_all_names()
        
        self.elevation_grid = self.generate_elevation()
        self.fill_types()

        self.routes = self.get_routes()
        self.trade_routes = self.get_trade_routes()
        self.land_trade_routes = self.get_trade_routes(overseas=False)

        self.trade_networks = self.get_trade_networks()
        self.land_trade_networks = self.get_trade_networks(overseas=False)

        # the states dictionary contains keys corresponding to times at which changes were made to the political landscape
        # other dictionaries should be updated in a similar way to save memory
        # to get information about the map, look through these for the latest time containing the desired information
        self.states = {}
        self.state_designation_counter = 0
        self.add_states(0) # time is the key of this dict so the history can be read through in a way

        self.rainfall = self.generate_rainfall()
        self.generate_resources()
        self.initiate_stockpiles()

    # types of resources, class variables, to be constructed in self.generate_resources()
    natural_resources = set()
    human_resources = set()
    technological_resources = set()

    def output(self, thing):
        if self.output_mode:
            self.output_string += str(thing) + "\n"
        else:
            return

        if len(self.output_string) > 1e4:
            self.flush_output()

    def flush_output(self, filepath="HistoryOutput.txt"):
        if self.output_mode:
            f = open(filepath, "a")
            f.write(self.output_string)
            f.close() # improve speed drastically by not calling this until the end
            self.output_string = ""
        else:
            pass

    def get_all_resource_names(self):
        return self.natural_resources | self.human_resources | self.technological_resources

    def generate_all_names(self):
        # new naming strategy: every point gets a name already, cities are named after their location, and states are named after some city in them
        # can add "-land" or such to states
        # this makes a finite number of state names, so if in the far future a new state gets an old name, can append numbers to it or something
        names = {}
        for r in range(self.r_size):
            for c in range(self.c_size):
                names[(r,c)] = self.generate_new_name()
        return names

    def generate_new_name(self):
        n = generate_name_global()
        # print("generated name",n)
        while n in self.used_names:
            # print("name {0} was duplicate".format(n))
            n = generate_name_global()
            # print("generated name",n)
        # print("name {0} was accepted".format(n))
        self.used_names.append(n)
        return n

    def generate_elevation(self):
        """
        this algorithm mandates an elevation shift of +1, 0, or -1 for a circle centered on each tile
        radii in [1 ... 1/2*min(x,y)]
        """

        result = [[0 for c in range(self.c_size)] for r in range(self.r_size)]

        # elevation grid creation part

        # first algorithm; quite slow for larger inputs, but distribution representation must retain conditionality of adjacent elevations
        # runs something like O(n^4) due to running through the whole grid for each spot, where n is grid side length
        # O(n^2) in number of spots; still bad
        # a lot of time was saved by rewriting grid_circle; self.generate_elevation() still runs in bad time, but practically I care less now

        for row in range(self.r_size):
            for col in range(self.c_size):
                #height_shift = random.choice([-1,0,1])
                height_shift = max(-2,min(2,random.normalvariate(0,1)))
                #height_shift = random.choice([-2,-1,0,1,2])
                #r = random.choice(range(1,math.ceil(0.5*min(self.r_size,self.c_size))))
                r = random.choice(range(1,8)) # make the radius distribution independent of map size so larger maps don't necessarily have more variance
                circle = grid_circle(self.r_size,self.c_size,row,col,r)
                for c_row in range(self.r_size):
                    for c_col_number in circle[c_row]:
                        result[c_row][c_col_number] += height_shift

        return result

    def fill_types(self):
        # area information filling part
        for row in range(self.r_size):
            for col in range(self.c_size):
                e = self.elevation_grid[row][col]
                if e < 0:
                    self.types[row][col].append("water")
                if e >= 0:
                    self.types[row][col].append("land")
                if e > 8:
                    self.types[row][col].append("mountain")
                    if is_local_maximum(self.elevation_grid, row, col):
                        self.types[row][col].append("peak")
                        if random.random() < 0.2:
                            self.types[row][col].append("volcano")
                            self.add_volcano(row, col)
                if is_on_shore(self.elevation_grid, row, col):
                    self.types[row][col].append("shore")
                    if random.random() < 0.1:
                        self.add_city(row,col)
                if is_local_minimum(self.elevation_grid, row, col):
                    if random.random() < 0.3:
                        self.add_city(row,col)

    types_dict = {
        "water":" ",
        "land":"-",
        "shore":"-", # can change later if want, but don't care about distinguishing from other land for now
        "mountain":"^",
        "peak":"A",
        "volcano":"V",
        "city":"c",
        "large_city":"C",
        "largest_city":"*"
    }

    def add_city(self,row,col):
        pop = population_dist() # screw with distribution later
        self.types[row][col].append("city")
        self.cities.append(City({"r":row, 
                            "c":col, 
                            "population":pop,
                            "name":self.names[(row,col)],
                            "map":self
                            }))
        
        if pop >= 1000:
            self.types[row][col].append("large_city")
        
        if pop > self.max_population:
            if self.largest_city != [-1,-1]:
                self.types[self.largest_city[0]][self.largest_city[1]].remove("largest_city")
            self.max_population = pop
            self.largest_city = [row,col]
            self.types[row][col].append("largest_city")

    def add_volcano(self, row, col):
        v = Volcano(Point((row, col)), self.names[(row, col)])
        self.volcanoes.append(v)
        print("Adding volcano {0} with average power {1:.2f}.".format(
            v.name, v.average_power))

    def get_route_between_cities(self, city1, city2):
        city1, city2 = sorted([city1, city2], key=lambda c: c.coordinates)

        return self.routes[city1.coordinates][city2.coordinates]

    def get_routes(self):
        """ All edges connecting nodes, not conditioned on navigability. """
        d = {}
        for city in sorted(self.cities, key=lambda c: c.coordinates):
            # maintain upper-triangle form of this matrix
            d[city] = {}
            for other_city in self.cities:
                if other_city is city:
                    d[city][other_city] = None
                elif other_city.coordinates < city.coordinates:
                    d[city][other_city] = d[other_city][city]
                else:
                    route = Route([city, other_city], self.r_size, self.c_size, self.types)
                    d[city][other_city] = route
        return d

    def get_trade_routes(self, overseas=True):
        # because lists are not hashable, cannot use them as keys
        # instead will just add lists with the coordinates of the cities to a list
        # "list of lists of lists"

        techs = {"ships"} if overseas else set()

        result = []
        for city in self.cities:
            for other_city in self.cities:
                route = self.routes[city][other_city]

                # # trade route is unacceptable if line between cities contains mountains (allow water since ships exist)
                # # will check types rather than elevations in case I later change the definitions of unacceptable types
                # add_pair = True
                # for type_lst in gl_types:
                #     if bad_types & set(type_lst) != set([]): # if their intersection is nonempty, the line contains a bad type
                #         add_pair = False
                #         break
                # if add_pair:
                #     pair = {city,other_city}
                #     if pair not in result:
                #         # don't duplicate routes by having the other direction already added
                #         result.append(pair)
                if route is not None and route.is_navigable(techs):
                    result.append(route)

        return result

    def add_states(self, t):
        # t argument is not really used here except to access the right key in self.states

        if t not in self.states:
            self.states[t] = {}

        for n in self.land_trade_networks:
            #n = sorted(sorted(self.land_trade_networks, key = lambda c: c.coordinates)[j])
            namesake_city = random.choice(list(n)) # get a random city in the network to name the state after
            name = self.names[namesake_city.coordinates]
            name = name+(" City-State" if len(n) == 1 else " Kingdom") # name of state
            display_index = len(name) # we will only display the name itself in pretty output, omitting the establishment date
            name += " est. {0}".format(t)
            display_name = name[:display_index]
            state_designation = self.state_designation_counter
            self.state_designation_counter += 1

            # all coordinates, for getting geographic center
            a = [i.r for i in n]
            b = [i.c for i in n]

            state = State({
                "name":name,
                "display_name":display_name,
                "network":set(n), # set of cities in network, converted from the frozenset representation in the attribute
                "population":sum([c.point.stockpiles["population"] for c in filter(lambda city: city.coordinates in n, self.cities)]),
                "geographic_center":((min(a)+max(a))/2.0,(min(b)+max(b))/2.0),
                "designation":state_designation,
                "map":self
            })

            self.states[t][name] = state
            for city in n:
                city.state = state
                state.cities.append(city)

    def get_states_as_of_time(self, t):
        # TODO
        return self.states[0].values()          

    def kill_everyone(self):
        for city in self.cities:
            city.point.stockpiles["population"] = 0
        self.max_population = 0
        self.largest_city = None

    def show_types(self): # depends on the type that should be shown having been added to the type list last
        s = ""
        s += ("/"+"-"*(self.c_size+2)+"\\") + "\n"
        s += ("|"+" "*(self.c_size+2)+"|") + "\n"
        for row in range(self.r_size):
            s += ("| "+"".join([self.types_dict[self.types[row][col][-1]] for col in range(self.c_size)])+" |") + "\n"
        s += ("|"+" "*(self.c_size+2)+"|") + "\n"
        s += ("\\"+"-"*(self.c_size+2)+"/") + "\n"
        self.output(s)

    def show_2d(self,style,t,cities=False,trade_routes="none",state_names=False):
        if style not in ["land_and_water","contour","land_contour"]:
            print("function show_2d in class Map: The style \"{0}\" was not found.".format(style))
            return

        # this is how this works:
        # r = [1,1,1,2,2,2,3,3,3,...]
        # c = [1,2,3,1,2,3,1,2,3,...]
        r = [r for c in range(self.c_size) for r in range(self.r_size)]
        c = [c for c in range(self.c_size) for r in range(self.r_size)]
        
        # let's construct zo more readably, bro
        # original: zo = [z for row in self.elevation_grid for z in row]
        zo = []
        # fucked up
        #for row in self.elevation_grid:
            #zo.extend(row)
        # have to make zo backwards because in r,c space we go downward but in x,y space we go upward
        for col in range(self.c_size):
            zo.extend([row[col] for row in self.elevation_grid[::-1]])

        if style=="contour":
            z = zo
        elif style=="land_contour":
            # map water points to min elevation
            z = [i for i in map(lambda x: min(zo) if x < 0 else 0.5+(x/2.0) if x < 1 else x, zo)]
            # that else 0.5 + x/2.0 if x < 1 part was supposed to make the islands with elevation close to zero more visible, but whatevs
        elif style=="land_and_water":
            # map each point to -1 for water or 1 for land
            z = [i for i in map(lambda x: 1.0*x/abs(x), zo)]

        r = np.array(r)
        c = np.array(c)
        z = np.array(z)

        ri = np.linspace(min(r)-1, max(r)+1)
        ci = np.linspace(min(c)-1, max(c)+1)
        C, R = np.meshgrid(ci, ri)
        Z = ml.griddata(c, r, z, ci, ri, interp="linear")

        # source: https://grantingram.wordpress.com/plotting-2d-unstructured-data-using-free-software/
        if style == "contour":
            levels = np.arange(math.floor(min(zo)),math.ceil(max(zo)),1)
        elif style == "land_contour":
            levels = np.array([math.floor(min(zo))]+[i for i in range(0,math.ceil(max(zo))+1)])
        else:
            levels = np.arange(0,math.ceil(max(zo)),1)
        #elif style == "land_and_water":
            #levels = [-1,1]
        csf = plt.contourf(C,R,Z,levels)
        cs = plt.contour(C,R,Z,levels,colors=("k"))
        #plt.clabel(cs)

        # trying to change the y-axis labels to correspond to row numbers for easier readability
        plt.yticks(range(self.r_size), [str(i) for i in range(self.r_size-1,-1,-1)])
        
        if cities:
            for city in self.cities:
                x = city.c
                y = self.r_size-city.r-1
                log_pop = int(math.log(max(1,city.point.stockpiles["population"]),2))*1.5
                plt.plot(x,y,"o",color="white",markersize=log_pop)


        if trade_routes != "none":
            routes = [] # don't want no local scope problems
            if trade_routes == "all":
                # routes = self.trade_routes
                all_routes = []
                for state in self.get_states_as_of_time(t):
                    all_routes.extend(state.get_internal_navigable_routes())
                routes = unique(all_routes)
            # elif trade_routes == "land":
            #     routes = self.land_trade_routes
            else:
                print("Map method show_2d: trade_routes argument invalid")
                stop()
            for route in routes:
                pair = list(route.cities)
                x1 = pair[0].x
                x2 = pair[1].x
                y1 = pair[0].y
                y2 = pair[1].y
                plt.plot([x1,x2],[y1,y2],linestyle="-",linewidth=3,color="black")
                plt.plot([x1,x2],[y1,y2],linestyle="-",linewidth=2,color="white")
        
        if state_names:
            these_states = sorted(self.states[t])
            x = [self.states[t][state]["geographic_center"][1] for state in these_states]
            y = [self.r_size - self.states[t][state]["geographic_center"][0] - 1 for state in these_states]
            names = [self.states[t][state]["name"] for state in these_states]
            plt.plot(x,y,"p",label=names) # make the markers transparent but the labels show up

        plt.show()


    def show_3d(self):
        # source: http://gis.stackexchange.com/questions/116319/

        r = [r for c in range(self.c_size) for r in range(self.r_size)]
        c = [c for c in range(self.c_size) for r in range(self.r_size)]
        
        # source on making flat list: http://stackoverflow.com/questions/952914/
        # original (not readable for debugging sidewaysness): z = [z for row in self.elevation_grid for z in row]
        z = []
        # fucked up
        #for row in self.elevation_grid:
            #z.extend(row)
        for col in range(self.c_size):
            z.extend([row[col] for row in self.elevation_grid[::-1]])

        r = np.array(r)
        c = np.array(c)
        z = np.array(z)
        
        ri = np.linspace(min(r), max(r))
        ci = np.linspace(min(c), max(c))
        C, R = np.meshgrid(ci, ri)
        Z = ml.griddata(c, r, z, ci, ri, interp="linear")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(C, R, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True)
        plt.show()

    def get_trade_networks(self,overseas=True):
        """
        Takes an option to include water routes or not, defaulting to overseas=True.
        Returns a set whose elements are sets of towns in the same network of trade routes.
        """
        if overseas:
            tr = [i for i in self.trade_routes] # in case of mutation
        else:
            tr = [i for i in self.land_trade_routes]
        networks = set()

        # the set of cities left is the cities which have not been added to networks
        cities_left = set(self.cities) #{city.coordinates for city in self.cities}
        
        while cities_left != set(): # make sure we get all the cities
            network = set()
            
            start_city = random.choice(list(cities_left))
            network.add(start_city)
            cities_left -= {start_city}
            
            # set of cities whose connections should still be added to network
            # the cities to consider will all already be in the network
            cities_to_consider = {start_city}
            
            # steps:
            # get a node from the set of cities to consider (seeded by each start city once a network is finished)
            # add that node to the network
            # remove that node from the set of cities to consider and the set of cities left
            # add that node's connections to the set of cities to consider

            while cities_to_consider != set():
                node = random.choice(list(cities_to_consider))
                network.add(node)
                cities_to_consider -= {node}
                cities_left -= {node}

                connections = {i.get_other_city(node) for i in filter(lambda route: node in route.cities, tr)} # syntax works as a set as well as as a list
                connections = {i for i in filter(lambda x: x in cities_left, connections)} # so we don't keep adding back cities that we already did
                #network |= connections
                cities_to_consider |= connections # "or equals" adds stuff using elementwise or operator (union); simil &= is intersection

            networks.add(frozenset(network))

        return networks

    def generate_rainfall(self):
        """
        Creates a matrix of rainfall amounts corresponding to each point in the grid.
        Operates by starting on the left of the map with equal default rainfall and working across to the right, rows independent for now.
        At each step, "picks up" a constant amount of rainfall into the amount left for dumping; dumps an amount proportional to the amount of land
        between sea level and some maximum height at that location.
        """

        result = []

        default = 5
        addend = 2
        max_height = 15.0

        for row in range(self.r_size):
            row_result = []
            reservoir = default
            for col in range(self.c_size):
                reservoir += addend
                el = max(0,min(max_height,self.elevation_grid[row][col]))
                ratio = float(el)/max_height
                rainfall = ratio * reservoir
                row_result.append(rainfall)
            result.append(row_result)

        return result

    def generate_resources(self):
        """
        Generates a grid whose elements are dictionaries containing amounts of certain resources.
        For randomly distributed resources that do not depend on other factors, uses the elevation grid generation algorithm (could be done faster).
        Some resources depend on elevation and proximity to water or volcanoes.
        """

        # resource_grid = [[{} for c in range(self.c_size)] for r in range(self.r_size)]

        # human resources

        # population; works differently from all the others
        for row in range(self.r_size):
            for col in range(self.c_size):
                if "city" in self.types[row][col]:
                    city = [i for i in filter(lambda c: c.coordinates == (row,col), self.cities)][0]
                    population = city.point.stockpiles["population"]
                    # population = city.point.resources["population"]
                else:
                    population = 0
                # resource_grid[row][col]["population"] = population
                self.points[(row,col)].add_resource_amount("population", population)
        self.human_resources.add("population")
        
        # natural resources

        # oil
        oil_eg = self.generate_elevation()
        for row in range(self.r_size):
            for col in range(self.c_size):
                oil = max(0, -10 - oil_eg[row][col])
                # resource_grid[row][col]["oil"] = oil
                self.points[(row,col)].add_resource_amount("oil", oil)
        self.natural_resources.add("oil")

        # fresh water
        for row in range(self.r_size):
            for col in range(self.c_size):
                # only give the freshwater resource to inland places with rain
                fw = self.rainfall[row][col]/5.0 if "shore" not in self.types[row][col] and "water" not in self.types[row][col] else 0
                # resource_grid[row][col]["fresh_water"] = fw
                self.points[(row,col)].add_resource_amount("fresh_water", fw)
        self.natural_resources.add("fresh_water")

        # rock
        for row in range(self.r_size):
            for col in range(self.c_size):
                # rock = self.elevation_grid[row][col]/3.0 if "mountain" in self.types[row][col] else 0
                rock = max(0, self.elevation_grid[row][col]/3.0)
                # resource_grid[row][col]["rock"] = rock
                self.points[(row,col)].add_resource_amount("rock", rock)
        self.natural_resources.add("rock")

        # metal
        metal_eg = self.generate_elevation()
        for row in range(self.r_size):
            for col in range(self.c_size):
                el = self.elevation_grid[row][col]
                metal = max(0, 20 + metal_eg[row][col]) if el >= 2 else 0
                # resource_grid[row][col]["metal"] = metal
                self.points[(row,col)].add_resource_amount("metal", metal)
        self.natural_resources.add("metal")

        # lava
        for row in range(self.r_size):
            for col in range(self.c_size):
                # more lava is readily available next to low-lying volcanoes
                neigh = neighbors(self.r_size,self.c_size,row,col,n=8)
                next_to_volcano = False
                for row_ in range(self.r_size):
                    for col_ in neigh[row_]:
                        if "volcano" in self.types[row_][col_]:
                            next_to_volcano = True
                lava = 20-self.elevation_grid[row][col] if next_to_volcano or random.random() < 0.05 else 0
                # resource_grid[row][col]["lava"] = lava
                self.points[(row,col)].add_resource_amount("lava", lava)
        self.natural_resources.add("lava")

        # grain
        for row in range(self.r_size):
            for col in range(self.c_size):
                el = self.elevation_grid[row][col]
                grain = (3.0-el)*self.rainfall[row][col]/3.0 if el >= 0 and el < 3 else 0
                # resource_grid[row][col]["grain"] = grain
                self.points[(row,col)].add_resource_amount("grain", grain)
        self.natural_resources.add("grain")

        # wood
        for row in range(self.r_size):
            for col in range(self.c_size):
                el = self.elevation_grid[row][col]
                wood = abs(6.0-el)*self.rainfall[row][col]/3.0 if el >= 0 and el < 12 else 0
                # resource_grid[row][col]["wood"] = wood
                self.points[(row,col)].add_resource_amount("wood", wood)
        self.natural_resources.add("wood")

        # technological resources
        # all initialized as zero because their development depends on other resources rather than a certain rate
        # their stockpiles will be built by self.develop_technology()

        for i in ["ships","guns","climbing_gear"]:
            for row in range(self.r_size):
                for col in range(self.c_size):
                    # resource_grid[row][col][i] = 0
                    self.points[(row,col)].add_resource_amount(i, 0)
            self.technological_resources.add(i)

        # return resource_grid

    def initiate_stockpiles(self):
        """
        Initiates the stockpile dictionary. Keys are tuples of coordinates (where cities exist), values are dictionaries of resource amounts.
        """

        # result = {}
        for rc in self.points:
            point = self.points[rc]
            for key in self.get_all_resource_names():
                point.add_stockpile_amount(key, 0)
            #     if key == "population":
            #         d[key] = city.population
            #     else:
            #         d[key] = 0
            # self.points[city.coordinates].stockpiles = d
        # return result

    def show_resources(self):
        for city in self.cities:
            r,c = city.coordinates
            self.output("{0} ({1}, {2}):".format(city.name,r,c))
            res = self.points[(r,c)].resources
            self.output({key:res[key] for key in sorted(res)})

    def show_stockpiles(self):
        for city in sorted(self.cities, key=lambda x: x.name):
            r,c = city.coordinates
            self.output("{0} ({1}, {2}) from {3}:".format(city.name, r, c, city.state.name))
            stock = self.points[(r,c)].stockpiles
            self.output({key:("%.2f" % stock[key]) for key in sorted(stock)})

    def mine_resource(self, resource, city): # for now, cities can only gather resources at their location, not nearby uninhabited places
        # add to the city's stockpile at the rate given by its amount of that resource per period
        if city.point.stockpiles["population"] < 1:
            return # cannot mine resource without people there to do it
        try:
            amount_to_add = max(0.01,min(100,random.normalvariate(1,0.3))) * city.point.resources[resource]
            city.point.add_stockpile_amount(resource, amount_to_add)
            taxation_factor = 1 # > 1 for compounded growth, < 1 for mean reversion; reasonable values are certainly in [0.99,1.01]
            taxed = random.random() < 1-1.0/(1 + city.point.stockpiles[resource])
            if taxed:
                city.point.stockpiles[resource] *= taxation_factor
            # self.output("Mined {0:.2f} {1} in {2}.".format(amount_to_add, resource, city.name) +
            #     (" Taxed at rate {0}.".format(taxation_factor) if taxed else "") +
            #     " New value: {0:.2f}.".format(city.point.stockpiles[resource]))
        except KeyError:
            self.output("resource entry {0} does not have resource {1}".format(city.point.resources, resource))
            # stop()
            city.point.resources[resource] = 0

    def mine_all_resources(self, city):
        #if city. != city.point.stockpiles["population"]:
            #raise Exception("The population of {0} is not working properly. city.population = {1}; city.point.stockpiles[\"population\"] = {2}".format(
                #city.name, city.population, city.point.stockpiles["population"]))
        for resource in self.get_all_resource_names():
            if True: #resource != "population":
                self.mine_resource(resource, city)
        city.point.stockpiles["population"] = int(city.point.stockpiles["population"])

    def trade_resource(self, resource, from_city, to_city):
        if to_city.point.stockpiles["population"] < 1:
            return

        from_amount = self.points[from_city.coordinates].stockpiles[resource]
        to_amount = self.points[to_city.coordinates].stockpiles[resource]
        #transfer_amount = 1 # constant rate
        transfer_amount = max(0,(from_amount-to_amount)/2.0) # rate conditional on "gradient steepness" between the cities

        self.points[from_city.coordinates].add_stockpile_amount(resource, -1*transfer_amount)
        self.points[to_city.coordinates].add_stockpile_amount(resource, transfer_amount)
        # self.output("Transferred {0:.2f} {1} from {2} to {3}. New amounts are {4:.2f}, {5:.2f}.".format(
        #     transfer_amount, resource, from_city.name, to_city.name,
        #     from_city.point.stockpiles[resource], to_city.point.stockpiles[resource]))

    def trade_all_resources(self, from_city, to_city):
        for resource in self.natural_resources | self.technological_resources:
            self.trade_resource(resource, from_city, to_city)

    def kill_people(self, city, number):
        number = int(number)
        pop = city.point.stockpiles["population"]
        number_killed = max(0, min(number,pop))
        city.point.add_stockpile_amount("population", -1*number_killed) # prevent resulting in negative populations
        if number_killed != 0:
            pass # self.output("Killed {0} people in {1}. Population now {2}.".format(number_killed, city.name, city.point.stockpiles["population"]))
        # if self.stockpiles[city.coordinates]["population"] <= 0:
        #     del city

    def move_people_out(self, from_city):
        """
        Chooses among the cities connected to the from_city, as well as the from_city itself, as optional places for people to move.
        Weights by amounts of resources, although the metric used could be changed.
        Parameter n is number of people to be moved at this time.
        Use this function as many times as desired to move people (does not have to be all people moving at once).
        """
        to_cities = from_city.get_trade_neighbors() + [from_city]
        d = {city:0 for city in to_cities}
        for city in to_cities:
            d[city] = city.get_stockpile_score()

        # for i in range(math.floor(from_city.point.stockpiles["population"]/10)):
        for i in range(10):
            chosen_city = weighted_choice(d)
            
            if chosen_city is None:
                raise RuntimeError("Map method move_people: no city was chosen to move people to")

            # self.move_people_between_cities(from_city,chosen_city,10)
            self.move_people_between_cities(from_city, chosen_city, math.ceil(from_city.point.stockpiles["population"]/10))

    def move_people_between_cities(self,from_city,to_city,n):
        n = max(n, min(from_city.point.stockpiles["population"], to_city.point.stockpiles["population"]))
        to_city.point.add_stockpile_amount("population", n)
        from_city.point.add_stockpile_amount("population", -1*n)
        if from_city != to_city:
            pass # self.output("Moved {n} people from {c_from} to {c_to}. Populations are now {p_from}, {p_to}.".format(
            #     n=n, c_from=from_city.name, c_to=to_city.name, p_from=from_city.point.stockpiles["population"], p_to=to_city.point.stockpiles["population"]))
        else:
            pass # self.output("{n} people decided to stay in {c_from}. Population is {p_from}.".format(
            #     n=n, c_from=from_city.name, p_from=from_city.point.stockpiles["population"]))

    def produce_resource(self, coords, output_name, input_dict):
        # sample of old code in case of climbing gear (for debugging the generalization)
        # n_climbing_gear = int(min(
        #     [self.stockpiles[coords][input_name]/float(input_dict[input_name]) for input_name in input_dict]
        #     #self.stockpiles[coords]["oil"]/10.0,
        #     #self.stockpiles[coords]["wood"]/20.0
        # ))
        # self.stockpiles[coords]["climbing_gear"] += n_climbing_gear
        # self.stockpiles[coords]["oil"] -= n_climbing_gear*10
        # self.stockpiles[coords]["wood"] -= n_climbing_gear*20

        n_produced = int(min(
            [self.points[coords].stockpiles[input_name]/float(input_dict[input_name]) for input_name in input_dict]
        ))
        self.points[coords].add_stockpile_amount(output_name, n_produced)
        for input_name in input_dict:
            self.points[coords].add_stockpile_amount(input_name, -1*n_produced*input_dict[input_name])

    def develop_technology(self, city): # please note that order matters; don't use all the wood on ships and then get no climbing gear!
        coords = city.coordinates

        # climbing gear # think of it as heavy machinery but made of wood so we don't always need metal
        self.produce_resource(coords, "climbing_gear", {"oil":10, "wood":20})

        # ships
        self.produce_resource(coords, "ships", {"wood":10, "metal":20})

        # guns
        self.produce_resource(coords, "guns", {"lava":3, "metal":5})

    def kill_people_at_random(self, city):
        kill_ratio = min(1, max(0, random.normalvariate(0, 0.05)))
        # people_to_kill = population_dist("death")*kill_ratio
        people_to_kill = city.point.stockpiles["population"] * kill_ratio
        self.kill_people(city, people_to_kill)

    def kill_people_from_volcano(self, volcano, power):
        temp_output = ""
        people_killed = 0
        temp_output += "Volcano {0} erupted with power {1:.2f}.".format(volcano.name, power) + "\n"
        for city in self.cities:
            distance = d(city.point, volcano.point)
            # if distance == 0:
            #     raise ValueError("City is in volcano. City {0} at {1}, volcano {2} at {3}.".format(
            #         city.name, city.point.coordinates, volcano.name, volcano.point.coordinates))
            effective_power = power *1.0/ (distance**2.0) if distance > 0 else float("inf")
            if effective_power == 0:
                continue
            effective_power_to_kill_half = 5
            denom = effective_power_to_kill_half / 2.0
            kill_ratio = max(0, 1 - 1.0/(effective_power *1.0/ denom)) # should kill everyone if and only if city is in volcano
            people_to_kill = int(city.point.stockpiles["population"] * kill_ratio)
            if people_to_kill > 0:
                self.kill_people(city, people_to_kill)
                people_killed += people_to_kill
                temp_output += "Effective power in {1} was {0:.2f}, killing {2} people. New population is {3}.".format(
                    effective_power, city.name, people_to_kill,
                    city.point.stockpiles["population"]) + "\n"

        if people_killed > 0:
            self.output(temp_output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=int, default=int(time.time()*10**4), help="Seed")
    parser.add_argument("-d", default="16,48", help="Map dimensions, separated by comma")
    parser.add_argument("-p", type=int, default=100, help="Number of periods to simulate")
    parser.add_argument("--no-show", dest="show_mode", action="store_false", help="Show plots, etc.; store_false")
    parser.add_argument("--no-output", dest="output_mode", action="store_false", help="Output to file; store_false")

    args = parser.parse_args()

    seed = args.s
    print("Seed used:",seed)
    random.seed(seed)
    map_dimensions_input = args.d
    map_dimensions = [int(i) for i in map_dimensions_input.split(",")]

    output_mode = args.output_mode
    if output_mode:
        open("HistoryOutput.txt", "w").close() # clear file

    M = Map(map_dimensions[0], map_dimensions[1], output_mode)

    n_p = args.p
    show_mode = args.show_mode

    if output_mode:
        M.show_types()
        #print("Populations:",sorted([city["population"] for city in M.cities]))
        #print(M.states)
        #print(M.rainfall)
        #M.show_resources()

    population_histories = {city.name:[] for city in M.cities}

    print("Simulating {0} periods.".format(n_p))
    M.show_2d(style="contour", cities=True, t=0, trade_routes="all")
    for t in range(n_p):
        M.output("\nCurrent period: {0}".format(t))
        print("Current period: {0}".format(t), end="\r")
        for volcano in M.volcanoes:
            power = int(volcano.erupt())
            if power > 0:
                M.kill_people_from_volcano(volcano, power)

        for city in sorted(M.cities, key = lambda x: x.name):
            M.mine_all_resources(city)
            M.develop_technology(city)
            M.move_people_out(city)
            M.kill_people_at_random(city)
            # to get the entire state for trade only along routes within the state
            for partner_city in city.trade_neighbors:
                M.trade_all_resources(city, partner_city)
            population_histories[city.name].append(city.point.stockpiles["population"])
            if city.point.stockpiles["population"] < 1:
                print("No one lives in {0}. Attempting to destroy it.".format(city.name))
                city.destroy()

        for state in M.get_states_as_of_time(t):
            state.go_on_conquest()

        M.show_2d(style="contour", cities=True, t=t, trade_routes="all")

        M.show_stockpiles()
    print()
    if output_mode:
        M.show_stockpiles()

    M.flush_output("HistoryOutput.txt")

    if show_mode:
        M.show_2d(style="contour", cities=True, t=0, trade_routes="all") # this is the one I like the most

    if show_mode:
        for city in M.cities:
            plt.plot(range(n_p),population_histories[city.name])
        plt.show()
        plt.close()
        world_population_history = [sum([population_histories[c.name][t] for c in M.cities]) for t in range(n_p)]
        plt.plot(world_population_history)
        plt.show()




# TODO
# (items labeled Ø are omitted for now, but may be done later if I feel like it)
#
# [X] add climates (just some naive way)
# - [X] add prevailing wind from some direction (let's just do left or right, why not just left for starters)
# - [X] dump rain on windward side, have dryness on leeward side
# - [X] assign resources to cities based on climate (oil completely random, fresh water in rainy areas, food in rainy low-lying areas, etc.)
# - [Ø] try to get rock and metal to appear more often for cities, check distribution of resources for reasonability
# [X] simulate trade and population dynamics
# - [X] resources in each city are produced at a certain rate and given to cities along routes at constant rate
# - [X] people move to cities with lots of resourcesad kak. (select from the neighbors, weighting by sum of resources or log-resources or something)
# - [X] people are also produced in a place at a rate correlating positively with food in the city (actually I just used randomly drawn birth rates)
# - [X] always have minimum population of 30, maybe start everyone with the same number once the dynamics work well (rather than starting Zipfian)
# - [Ø] cities are abandoned if everyone leaves, but they can remain as "stops" in the trade network afterwards
#       (this is most easily achieved by making population just stop at 0, making all mining/crafting cease, and leaving the city there)
# [X] assign each city a time to develop the technology to trade over water and then mountains (the kingdom it is in at the time can then use the tech)
# - [X] create types of technology (ships, guns, mountain-crossing gear, etc.) that is developed in a city once a certain amount of resources is obtained
# - [X] implement the ability to use this technology for trade and exploration/conquest, preferably by exploration first, then add new cities to the network
# [X] simulate exploration and conquest
# - [X] states launch exploration from their cities to reachable places (if hit city, battle)
# - [X] states with more tech reach the ones that are behind and take over their cities based on who has more of some kind of military resource
# - [X] conquered cities are removed from old state's trade network and added to new state's; this may cause old state to split in some cases
# - [ ] new cities can be founded in places with good resources (use the resource score function for this)
# [X] simulate volcanic eruptions and corresponding decreases in population
# - [ ] short-term climate change
# [ ] generate account of the history of the world (under constant development)
# - [X] be able to view graphs of cities' resources over time, make this an instance method
# [ ] display map, resource maps, etc. in a pygame window, clickable or easily keyboard-navigable between various maps and options
# [ ] instead of trading everything uniformly between connected cities, have markets where cities price goods based on supply, etc.
# [ ] cities evaluate happiness on per-capita resources and rebel against the state if they are unhappy

# FIXME
#
# [ ] allow for removal of cities from the entire map if they reach a population of zero (just to eliminate unnecessary computation)
# [ ] change "City-State" to "Kingdom" and vice versa when states shrink to or grow from one member city














