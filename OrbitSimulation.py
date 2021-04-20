import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import itertools


G = 6.67430e-11  # SI units: m^3 kg^-1 s^-2


class Body:
    def __init__(self, name, radius, mass, starting_orbital_radius, starting_orbital_speed, starting_orbital_phase=None):
        self.name = name
        self.radius = radius
        self.mass = mass
        self.starting_orbital_radius = starting_orbital_radius
        self.starting_orbital_phase = starting_orbital_phase if starting_orbital_phase is not None else random.uniform(0, 2*np.pi)

        self.starting_x = self.starting_orbital_radius * np.cos(self.starting_orbital_phase)
        self.starting_y = self.starting_orbital_radius * np.sin(self.starting_orbital_phase)
        self.starting_z = 0
        self.position = np.array([self.starting_x, self.starting_y, self.starting_z])  # don't modify in place, just replace it with new

        self.starting_orbital_speed = starting_orbital_speed  # assume tangential counterclockwise
        self.starting_velocity_angle = self.starting_orbital_phase + np.pi/2  # tangent counterclockwise
        self.starting_vx = self.starting_orbital_speed * np.cos(self.starting_velocity_angle)
        self.starting_vy = self.starting_orbital_speed * np.sin(self.starting_velocity_angle)
        self.starting_vz = 0
        self.velocity = np.array([self.starting_vx, self.starting_vy, self.starting_vz])  # don't modify in place, just replace it with new
        print(f"{self} initialized with position: {self.position}, velocity: {self.velocity}")

    def total_force_from(self, other_bodies):
        f = 0
        for b in other_bodies:
            f += self.single_force_from(b)
        return f

    def single_force_from(self, other):
        xyz0 = self.position
        xyz1 = other.position
        r = xyz1 - xyz0  # in direction of the other
        r_mag = np.linalg.norm(r)
        assert r_mag > 0, f"zero radius vector found in single force from {other} on {self}: {r}"
        r_hat = r / r_mag
        m0 = self.mass
        m1 = other.mass
        return G*m0*m1/(r_mag**2) * r_hat

    def move(self, other_bodies, time_step):
        f = self.total_force_from(other_bodies)
        a = f/self.mass
        v = self.velocity
        x = self.position
        new_v = v + a*time_step
        average_v = (v + new_v)/2
        new_x = x + average_v*time_step
        self.position = new_x
        self.velocity = new_v

    def collides(self, other):
        r = self.distance_to(other)
        min_r_allowed = self.radius + other.radius  # just touching each other
        return r <= min_r_allowed

    def distance_to(self, other):
        xyz0 = self.position
        xyz1 = other.position
        r = np.linalg.norm(xyz0-xyz1)
        return r

    def __repr__(self):
        return "<Body {}>".format(self.name)


def get_gravitational_acceleration(parent, radius):
    m0 = parent.mass
    r = radius
    return G*m0/(r**2)


def get_circular_orbital_speed(parent, orbital_radius):
    # acceleration is v^2 / r, should equal g
    g = get_gravitational_acceleration(parent, orbital_radius)
    v = math.sqrt(g*orbital_radius)
    return v


def any_collision_occurs(bodies):
    for a, b in itertools.combinations(bodies, 2):
        if a.collides(b):
            assert b.collides(a)
            print("{} and {} collided!".format(a, b))
            return True
        else:
            assert not b.collides(a)  # commutative
    return False


def max_distance(bodies):
    d = 0
    for a, b in itertools.combinations(bodies, 2):
        d1 = a.distance_to(b)
        d = max(d, d1)
    return d


def move_bodies(bodies, time_step):
    for body in bodies:
        other_bodies = [b for b in bodies if b is not body]
        body.move(other_bodies, time_step)


def plot_motion(bodies, time_step, plot_every_n_steps):
    plt.ion()
    fignum = plt.gcf().number
    colors = ["blue", "red", "orange", "green"]

    i = 0
    d_max = 0
    while True:
        if i % plot_every_n_steps == 0:
            print(f"iteration {i} for time step {time_step} = time {i*time_step}")
            plt.gcf().clear()
    
            d_max = max(d_max, max_distance(bodies))  # keep running maximum so plot doesn't grow and shrink in cycles
            lim = 1.1 * d_max
            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)
    
            for body, color in zip(bodies, colors):
                x,y,z = body.position
                plt.scatter(x, y, c=color)
            plt.draw()
            plt.pause(0.01)
    
            if any_collision_occurs(bodies):
                print("collision occurred!")
                input("press enter to continue")
            if not plt.fignum_exists(fignum):
                print("user closed plot; exiting")
                plt.ioff()
                return

        move_bodies(bodies, time_step)
        i += 1

    plt.ioff()


if __name__ == "__main__":
    cada_ii = Body("Cada II", radius=1.35e7, mass=4.89e25, starting_orbital_radius=0, starting_orbital_speed=0)

    impa_orbital_radius = 182405000
    lamdo_orbital_radius = 350671000
    hartha_orbital_radius = 683388000
    impa_orbital_speed = -1 * get_circular_orbital_speed(cada_ii, impa_orbital_radius)
    lamdo_orbital_speed = get_circular_orbital_speed(cada_ii, lamdo_orbital_radius)
    hartha_orbital_speed = get_circular_orbital_speed(cada_ii, hartha_orbital_radius)

    impa = Body("Impa", radius=289105, mass=3.425e20, starting_orbital_radius=impa_orbital_radius, starting_orbital_speed=impa_orbital_speed)
    lamdo = Body("Lamdo", radius=2496177, mass=1.07693e23, starting_orbital_radius=lamdo_orbital_radius, starting_orbital_speed=lamdo_orbital_speed)
    hartha = Body("Hartha", radius=672665, mass=3.66286e21, starting_orbital_radius=hartha_orbital_radius, starting_orbital_speed=hartha_orbital_speed)
    bodies = [cada_ii, impa, lamdo, hartha]

    plot_motion(bodies, time_step=10, plot_every_n_steps=10000)
