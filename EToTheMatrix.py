# from 3Blue1Brown video

import numpy as np


def e_taylor_sum(x, convergence_threshold=1e-6):
    s = 0.0
    n = 0
    g = power_generator(x)
    while True:
        # next_term = e_taylor_term(x, n)  # wasting computation by throwing away previous power results and redoing it
        next_power = next(g)
        next_term = next_power / np.math.factorial(n)
        # print("next term: {}".format(next_term))
        s += next_term.astype(float)
        # print("s: {}".format(s))
        n += 1
        if not is_finite(s):
            raise RuntimeError("failed to converge")
        if abs_general(next_term) < convergence_threshold:
            return s
        if n > 1000:
            raise RuntimeError("failed to converge")


def is_finite(x):
    if type(x) is np.ndarray:
        return np.isfinite(x).all()
    return np.isfinite(x)


def e_taylor_term(x, n):
    return power(x, n) / np.math.factorial(n)


def abs_general(x):
    if type(x) is np.ndarray:
        return np.linalg.norm(x)
    return abs(x)


def power(x, n):
    if type(x) is np.ndarray:
        s1, s2 = x.shape
        assert s1 == s2
        if n == 0:
            return np.identity(s1)
        elif n == 1:
            return x
        else:
            previous = power(x, n-1)
            return np.matmul(x, previous)
    return x ** n


def power_generator(x):
    if type(x) is np.ndarray:
        s1, s2 = x.shape
        assert s1 == s2
        I = np.identity(s1)
        yield I
        m = I
        while True:
            m = np.matmul(x, m).astype(float)
            yield m
    else:
        I = 1
        yield I
        m = I
        while True:
            m *= x
            yield m


if __name__ == "__main__":
    a = np.random.randint(-10, 10, (2,2))
    # a = np.array([[0, -np.pi], [np.pi, 0]])
    print("matrix:")
    print(a)
    s = e_taylor_sum(a)
    print("exp(matrix):")
    print(s)
