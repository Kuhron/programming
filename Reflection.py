import numpy as np
import matplotlib.pyplot as plt


class NormalVectorField:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.coordinates = np.array([[(r, c) for c in range(self.n_cols)] for r in range(self.n_rows)])
        self.rows = self.coordinates[:, :, 0]
        self.cols = self.coordinates[:, :, 1]
        self.vectors = np.zeros((n_rows, n_cols, 3))

    def get_wave_field(self):
        # generate a single wave in one direction, random theta
        # return normal vectors resulting from that surface

        amplitude = np.random.uniform(3, 10)
        frequency = np.random.uniform(3, 10)
        theta = np.random.uniform(0, np.pi)
        row_phase, col_phase = np.random.uniform(0, 2*np.pi, (2,))

        # transform each point to its "x" coordinate along the direction of wave propagation
        # just take a simple sin of that as the DERIVATIVE of height w.r.t. "x"
        # get normal vector's theta from height-axis, find normal vector components
        # all of this should be done in one numpy function, but can use auxiliary lambdas
        # return the vector field itself

        # later can run this function multiple times and take the sum as self.vectors