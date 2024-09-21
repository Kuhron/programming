# get the correlation matrix of each pair of dimensions of data

import numpy as np
import random


def get_correlation_cost(points):
    # assume it's an array of column vectors, one for each point
    corr = np.corrcoef(points)
    cost = 0
    n_dim, n_points = points.shape
    for i in range(n_dim - 1):
        for j in range(i+1, n_dim):
            cost += corr[i][j] ** 2
    return cost


def find_cost_minimizing_transformation_brute_force(points):
    cost = get_correlation_cost(points)
    n_dim, n_points = points.shape
    steps_without_improvement = 0
    last_improvement_size = None
    best_m = None
    while True:
        m = np.random.uniform(-1, 1, (n_dim, n_dim))
        m /= (abs(np.linalg.det(m)) ** (1/n_dim))  # normalize so det is 1 or -1
        assert np.isclose(abs(np.linalg.det(m)), 1, atol=1e-3)
        p2 = m @ points
        new_cost = get_correlation_cost(p2)
        improvement = new_cost - cost
        if improvement < 0:
            # cost has gone down
            last_improvement_size = abs(improvement)
            print(f"{last_improvement_size = :.4f}")
            steps_without_improvement = 0
            cost = new_cost
            best_m = m
        else:
            steps_without_improvement += 1
            print(f"{steps_without_improvement = }", end="\r")
            if steps_without_improvement >= 10000:
                print()  # so there's not a carriage return still there
                return best_m


def normalize_variances(points):
    # make var=stdev=1 on all axes
    n_dim, n_points = points.shape
    for i in range(n_dim):
        vals = points[i, :]
        stdev = np.std(vals)
        points[i, :] /= stdev
    return points


if __name__ == "__main__":
    # TODO:
    # - it turns out that there's a solution where simply rotating the dataset gives zero cost (at least in 2D and I suspect in other dims too)
    # -- this is NOT what I want; I want a "prying-apart" transformation where lines sort of parallel to the correlation line are used as the new basis


    n_dim = 3
    n_points = 100
    points = np.zeros((n_dim, n_points))  # use column vectors for points so the transformation can just be applied as M @ points and the corrcoef function can be called on points rather than points.T (the array `points` is a list whose elements each correspond to the values on one dimension)
    r = lambda: np.random.random((n_dim,))
    points[:, 0] = r()

    for i in range(1, n_points):
        p0 = points[:, i-1]
        p1 = points[:, (i-1)//2]
        p = (p0 + p1)/2 + r()
        points[:, i] = p

    for i in range(n_dim):
        points[i, :] *= random.uniform(-3, 3)

    print(np.corrcoef(points))
    print(get_correlation_cost(points))

    m = find_cost_minimizing_transformation_brute_force(points)
    print(m)
    p2 = m @ points
    print(np.corrcoef(p2))
    print(get_correlation_cost(p2))

