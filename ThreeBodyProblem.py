import math
import random

import matplotlib.pyplot as plt


class Star:
    def __init__(self, mass, phase_vector):
        self.mass = mass
        self.update_phase_vector(phase_vector)

    def update_phase_vector(self, phase_vector):
        self.phase_vector = phase_vector
        self.location, self.velocity, self.acceleration = phase_vector.derivatives[:3]


class PhaseVector:
    def __init__(self, derivatives):
        self.derivatives = derivatives
        self.location, self.velocity, self.acceleration = derivatives[:3]


class Point:
    def __init__(self, coordinates):
        self.dimensions = len(coordinates)
        self.coordinates = coordinates
        self.magnitude = get_euclidean_distance(get_zero_point(self.dimensions),self) if not all([abs(i) < 10**-9 for i in self.coordinates]) else 0
        self.unit_vector_coordinates = tuple([(i * 1.0 / self.magnitude) if self.magnitude != 0 else 0
            for i in self.coordinates])


class System:
    def __init__(self, dimensions, n_bodies, G):
        self.dimensions = dimensions
        self.n_bodies = n_bodies
        self.G = G
        self.stars = [get_stationary_star(dimensions) for i in range(n_bodies)]
        plt.scatter(range(self.n_bodies), sorted([i.mass for i in self.stars]))
        plt.show()
        plt.close()

    def get_gravity_matrix(self, G):
        return [[get_gravitational_force_vector(star_r, star_c, G) for star_c in self.stars] for star_r in self.stars]

    def move_stars(self, G):
        gm = self.get_gravity_matrix(G)

        for i in range(self.n_bodies):
            a_0 = self.stars[i].acceleration
            v_0 = self.stars[i].velocity
            x_0 = self.stars[i].location

            a_1 = vector_list_sum(gm[i])
            v_1 = add_vectors(v_0, a_1)
            x_1 = add_vectors(x_0, v_1)

            self.stars[i].update_phase_vector(PhaseVector(tuple([x_1, v_1, a_1])))

    def simulate(self, n_periods, period_length, show_intermediate_plots):
        G_per_period = self.G * (period_length)**2
        tenth_of_the_way = int(n_periods / 10)
        star_speeds = [[self.stars[i].velocity.magnitude] for i in range(self.n_bodies)]
        star_locations = [[self.stars[i].location] for i in range(self.n_bodies)]
        #star_distances_from_origin = [[self.stars[i].location.magnitude] for i in range(self.n_bodies)]
        #star_x_coordinates = [[self.stars[i].location.coordinates[0]] for i in range(self.n_bodies)]
        for t in range(n_periods):
            self.move_stars(G_per_period)
            for i in range(self.n_bodies):
                star_speeds[i].append(self.stars[i].velocity.magnitude)
                star_locations[i].append(self.stars[i].location)

            if show_intermediate_plots and self.dimensions == 2 and t % tenth_of_the_way == 0:
                x = [self.stars[i].location.coordinates[0] for i in range(self.n_bodies)]
                y = [self.stars[i].location.coordinates[1] for i in range(self.n_bodies)]
                plt.scatter(x,y)
                plt.show()
                plt.close()

        plot_series_set([[i.magnitude for i in star] for star in star_locations],"Distance from origin")
        if self.dimensions <= 4:
            for dim in range(self.dimensions):
                plot_series_set([[i.coordinates[dim] for i in star] for star in star_locations],"Coordinate in {0} dimension".format(ordinal(dim+1)))


def ordinal(n):
    n = str(int(n))
    if n[-1] == "1":
        return n[:-1] + "1st"
    elif n[-1] == "2":
        return n[:-1] + "2nd"
    elif n[-1] == "3":
        return n[:-1] + "3rd"
    else:
        return n + "th"

def plot_series_set(series_set, title):
    for series in series_set:
        plt.plot(series)
    plt.title(title)
    plt.show()
    plt.close()

def get_stationary_star(dimensions):
    DIMENSIONS = dimensions
    DERIVATIVES = 2
    random_point = get_random_point(DIMENSIONS)
    stationary_derivatives = [Point(tuple([0 for i in range(DIMENSIONS)])) for i in range(DERIVATIVES)]
    s = Star(mass = random.paretovariate(0.7), phase_vector = PhaseVector([random_point] + stationary_derivatives))
    return s

def get_zero_point(n_dimensions):
    return Point(tuple([0 for i in range(n_dimensions)]))

def get_random_point(n_dimensions):
    return Point(tuple([random.uniform(-100,100) for i in range(n_dimensions)]))

def get_random_factor():
    return random.uniform(0.99,1/0.99)

def get_euclidean_distance(point_a, point_b):
    if point_a.dimensions != point_b.dimensions:
        raise ValueError("Points must have same number of dimensions")
    return math.sqrt(sum([
        (point_a.coordinates[dim] - point_b.coordinates[dim])**2 for dim in range(point_a.dimensions)]))

def get_gravitational_force_magnitude(star_a, star_b, G):
    r = get_euclidean_distance(star_a.location, star_b.location)
    return (-1 * G * star_a.mass * star_b.mass * 1.0 / (r**2)) if r != 0 else 0

def get_gravitational_force_vector(star_ref, star_applying_force, G):
    F = get_gravitational_force_magnitude(star_ref, star_applying_force, G)
    S = get_separation_vector(star_ref, star_applying_force)
    return get_vector(F, S)

def get_acceleration_magnitude(force_magnitude, mass):
    return force_magnitude * 1.0 / mass

def get_distance_travelled(phase_vector, time_difference):
    x, v, a = phase_vector.derivatives
    t = time_difference
    return (1.0/2 * a * t**2) + (v * t)

def get_displacement_vector(movement_phase_vector, time_difference):
    d = get_distance_travelled(movement_phase_vector, time_difference)
    return get_vector(d, movement_phase_vector.unit_vector_coordinates)

def get_separation_vector(star_ref, star_to):
    return subtract_vectors(star_ref.location, star_to.location)

def get_vector(magnitude, unit_vector_coordinates):
    d = magnitude
    return Point(tuple([d*i for i in unit_vector_coordinates.coordinates]))

def add_vectors(vector_a, vector_b):
    if vector_a.dimensions != vector_b.dimensions:
        raise ValueError("Vectors must have same number of dimensions")
    coordinates = tuple([vector_a.coordinates[dim] + vector_b.coordinates[dim] for dim in range(vector_a.dimensions)])
    return Point(coordinates)

def subtract_vectors(vector_a, vector_b):
    if vector_a.dimensions != vector_b.dimensions:
        raise ValueError("Vectors must have same number of dimensions")
    coordinates = tuple([vector_a.coordinates[dim] - vector_b.coordinates[dim] for dim in range(vector_a.dimensions)])
    return Point(coordinates)

def vector_list_sum(vector_list):
    p = vector_list[0]
    for i in range(1, len(vector_list)):
        p = add_vectors(p, vector_list[i])
    return p


#random.seed(4)
a = System(dimensions=3, n_bodies=6, G=4)
a.simulate(n_periods=10000, period_length=0.1, show_intermediate_plots=True)


