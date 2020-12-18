import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix



class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity
    
    @staticmethod
    def get_random_particle():
        mass = np.random.uniform(0, 100)
        position = np.random.uniform(-1, 1, (2,))
        # velocity = np.random.uniform(-1, 1, (2,))
        velocity = np.zeros((2,))
        return Particle(mass, position, velocity)

    def move(self, displacement):
        self.position += displacement

    def accelerate(self, acceleration):
        self.velocity += acceleration


class UniverseState:
    def __init__(self, particles):
        self.particles = particles

    @staticmethod
    def get_random_initial_state(n_particles):
        particles = []
        for i in range(n_particles):
            p = Particle.get_random_particle()
            particles.append(p)
        return UniverseState(particles)

    def get_masses(self):
        return np.array([p.mass for p in self.particles])

    def get_positions(self):
        return np.array([p.position for p in self.particles])

    def get_distance_matrix(self):
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

    def get_accelerations(self):
        distances = self.get_distance_matrix()
        radial_vectors = self.get_radial_vectors_between_particles()
        r_squareds = distances ** 2
        mass_products = self.get_mass_product_matrix()
        G = 1e-6
        force_magnitudes = np.where(distances == 0, 0, G * mass_products / r_squareds)
        forces = force_magnitudes[:, :, None] * -1 * radial_vectors  # for each cell in (n,n) grid, multiply 2d vector by 1d magnitude, so need to broadcast (n,n) magnitude array to (n,n,1)
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

    def evolve(self):
        # I choose to change the velocities first, then step-move the particles with their new velocities
        accelerations = self.get_accelerations()
        self.accelerate_particles(accelerations)
        self.move_particles()

    def plot(self, restrict_to_original_box=False):
        positions = self.get_positions()
        plt.scatter(*positions.T)
        if restrict_to_original_box:
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)



if __name__ == "__main__":
    plt.ion()
    fignum = plt.gcf().number  # use to determine if user has closed plot

    n_particles = 100
    state = UniverseState.get_random_initial_state(n_particles)
    while True:
        if not plt.fignum_exists(fignum):
            print("user closed plot; exiting")
            break
        state.evolve()
        plt.gcf().clear()
        state.plot(restrict_to_original_box=False)
        plt.draw()
        plt.pause(0.001)

