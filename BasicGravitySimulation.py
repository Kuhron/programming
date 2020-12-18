import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix



class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.color = self.get_color()
    
    @staticmethod
    def get_random_particle(max_xy=1, allow_negative_mass=True):
        if allow_negative_mass:
            mass_sign = random.choice([-1, 1])
        else:
            mass_sign = 1
        mass = 10 * np.random.pareto(a=1) * mass_sign
        position = np.random.uniform(-max_xy, max_xy, (2,))
        # velocity = np.random.uniform(-1, 1, (2,))
        velocity = np.zeros((2,))
        return Particle(mass, position, velocity)

    def get_color(self):
        # if negative mass, make it more red
        # if self.mass >= 0:
        #     red = random.uniform(0, 0.3)
        # else:
        #     red = random.uniform(0.7, 1)
        # green, blue = np.random.uniform(0, 1, (2,))
        # return (red, green, blue)
        return (1, 0, 0) if self.mass < 0 else (0, 0, 1)


    def move(self, displacement):
        self.position += displacement

    def accelerate(self, acceleration):
        self.velocity += acceleration


class UniverseState:
    def __init__(self, particles):
        self.particles = particles
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

    @staticmethod
    def get_random_initial_state(n_particles, allow_negative_mass=True):
        particles = []
        for i in range(n_particles):
            p = Particle.get_random_particle(allow_negative_mass=allow_negative_mass)
            particles.append(p)
        return UniverseState(particles)

    def get_masses(self):
        return np.array([p.mass for p in self.particles])

    def get_positions(self):
        positions = np.array([p.position for p in self.particles])
        xs = positions[:,0]  # all points, first item in 2d vector for that point
        ys = positions[:,1]
        self.min_x = min(self.min_x, min(xs))
        self.max_x = max(self.max_x, max(xs))
        self.min_y = min(self.min_y, min(ys))
        self.max_y = max(self.max_y, max(ys))

        return positions

    def get_distance_matrix(self):
        if len(self.particles) == 0:
            raise Exception("can't get distance matrix from zero particles")
        x = self.get_positions()
        n_particles, n_dim = x.shape
        assert n_particles == len(self.particles)
        assert n_dim == 2
        m = distance_matrix(x, x)
        assert m.shape == (n_particles, n_particles)
        return m

    def get_displacement_matrix(self):
        positions = self.get_positions()
        diff = lambda x, y: x-y
        positions_broadcast = positions[:,None]
        # about broadcasting, see e.g.:
        # https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
        # https://stackoverflow.com/questions/42542154/user-defined-function-for-numpy-outer
        # https://stackoverflow.com/questions/21226610/numpy-quirk-apply-function-to-all-pairs-of-two-1d-arrays-to-get-one-2d-array
        displacements = diff(positions, positions_broadcast)
        return displacements

    def get_radial_vectors_between_particles(self):
        displacements = self.get_displacement_matrix()
        n = len(self.particles)
        assert displacements.shape == (n, n, 2)  # 2d displacement vector for each pair of particles
        magnitudes = np.linalg.norm(displacements, axis=2)  # do the norm along the number 2 axis (the one containing the x,y vector), for each point in the sub-array without that axis (here, the n by n grid of point pairs)
        magnitudes_broadcast = magnitudes[:, :, None]  # turns it from (n,n) to (n,n,1) so can do the division of (2,) vector by (1,) scalar for each pair in the (n,n) grid
        radial_vectors = displacements / magnitudes_broadcast
        radial_vectors = np.where(magnitudes[:,:,None] == 0, 0, radial_vectors)
        return radial_vectors

    def get_mass_product_matrix(self):
        a = self.get_masses()
        assert a.shape == (len(self.particles),), "{} particles but mass shape is {}".format(len(self.particles), a.shape)
        return np.outer(a, a)

    def move_particles(self):
        for p in self.particles:
            p.move(p.velocity)

    def accelerate_particles(self, accelerations):
        for p, a in zip(self.particles, accelerations):
            p.accelerate(a)

    def get_accelerations(self, gravity_power=2, big_g=1e-3):
        distances = self.get_distance_matrix()
        radial_vectors = self.get_radial_vectors_between_particles()
        force_factors = 1 / (distances ** gravity_power)

        # make a possibly interesting modification: force goes up dramatically again once objects are too far away, keeping the system bound
        # distant_distance = 10
        # get_distant_attractive_force = lambda d: np.maximum(0, d - distant_distance) ** 2  # have to use maximum, not max, to compare array and scalar and get array output
        # force_factors += get_distant_attractive_force(distances)


        mass_products = self.get_mass_product_matrix()
        gravity_direction = -1  # -1 for attractive, 1 for repulsive
        force_magnitudes = np.where(distances == 0, 0, big_g * mass_products * force_factors)
        forces = force_magnitudes[:, :, None] * gravity_direction * radial_vectors  # for each cell in (n,n) grid, multiply 2d vector by 1d magnitude, so need to broadcast (n,n) magnitude array to (n,n,1)
        forces_per_particle = forces.sum(axis=0)  # arr.sum(**kwargs) is same as np.sum(arr, **kwargs)
        masses = self.get_masses()
        accelerations = forces_per_particle / masses[:, None]
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

        debug = False
        if debug:
            print("\n\n-- getting accelerations")
            print("positions:\n{}".format(self.get_positions()))
            print("distances:\n{}".format(distances))
            print("radial vectors:\n{}".format(radial_vectors))
            print("force magnitudes:\n{}".format(force_magnitudes))
            print("forces:\n{}".format(forces))
            print("masses:\n{}".format(masses))
            print("accelerations:\n{}".format(accelerations))
            print("acceleration magnitudes:\n{}".format(acceleration_magnitudes))
            # input("a")

        return accelerations

    def evolve(self, gravity_power=2, big_g=1e-3):
        # I choose to change the velocities first, then step-move the particles with their new velocities
        accelerations = self.get_accelerations(gravity_power=gravity_power, big_g=big_g)
        self.accelerate_particles(accelerations)
        self.move_particles()

        n = len(self.particles)
        assert n > 0
        # get rid of ones that are going too far away
        self.prune(min_x=-10, max_x=10, min_y=-10, max_y=10)
        self.add_new_particles(n)

    def prune(self, min_x, max_x, min_y, max_y):
        particles = []
        for p in self.particles:
            x, y = p.position
            if min_x <= x <= max_x and min_y <= y <= max_y:
                # keep it, it's within range
                particles.append(p)
        self.particles = particles

    def add_new_particles(self, n_total_to_achieve):
        n = len(self.particles)
        n_to_make = n_total_to_achieve - n
        if n_to_make <= 0:
            return
        new_particles = [Particle.get_random_particle(max_xy=10) for i in range(n_to_make)]
        self.particles += new_particles
        assert len(self.particles) == n_total_to_achieve

    def plot(self, restrict_to_original_box=False, restrict_to_record_box=False):
        positions = self.get_positions()
        masses = self.get_masses()
        point_sizes = np.maximum(5, abs(masses) ** (2/3))  # point size is area of the dot on the plot, not radius
        colors = [p.color for p in self.particles]
        plt.scatter(*positions.T, s=point_sizes, c=colors)
        if restrict_to_original_box:
            assert not restrict_to_record_box
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
        elif restrict_to_record_box:
            assert not restrict_to_original_box
            plt.xlim(self.min_x, self.max_x)
            plt.ylim(self.min_y, self.max_y)



if __name__ == "__main__":
    plt.ion()
    fignum = plt.gcf().number  # use to determine if user has closed plot

    n_particles = 2
    allow_negative_mass = False
    gravity_power = 1/3  # 2 is normal (1/r^p)
    big_g = 1e-6
    plot_every_n_steps = 100
    state = UniverseState.get_random_initial_state(n_particles, allow_negative_mass=allow_negative_mass)

    step_i = 0
    if n_particles == 2:
        distances = []
    while True:
        if not plt.fignum_exists(fignum):
            print("user closed plot; exiting")
            break
        state.evolve(gravity_power=gravity_power, big_g=big_g)
        if n_particles == 2:
            assert len(state.particles) == 2
            distance = state.get_distance_matrix()[0, 1]
            distances.append(distance)
        if step_i % plot_every_n_steps == 0:
            print("step {}".format(step_i))
            plt.gcf().clear()
            state.plot(restrict_to_original_box=False, restrict_to_record_box=True)
            plt.draw()
            plt.pause(0.001)
        step_i += 1
    plt.ioff()  # how to close the ion()

    if n_particles == 2:
        plt.plot(distances)
        plt.show()
