import random

import numpy as np
import matplotlib.pyplot as plt


def get_parametric_function():
    def get_f():
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        c = random.uniform(-np.pi, np.pi)
        d = random.uniform(-10, 10)
        e = random.uniform(-np.pi, np.pi)
        p_pows = np.arange(0, 3.5, 0.1)
        c_pows = [0 if random.random() < 0.8 else random.normalvariate(0, 2**-p) for p in p_pows]
        a_pows = [random.uniform(-2, 2) for p in p_pows]
        b_pows = [random.uniform(-20, 20) for p in p_pows]
        pows = zip(p_pows, a_pows, b_pows, c_pows)
        return lambda t: a * t + b * np.sin(t + c) + d * np.cos(t + e) + sum(c*np.power(a*(t+0j)+b, p).real for p, a, b, c in pows)

    x = get_f()
    y = get_f()
    # z = get_f()

    return lambda t: (x(t), y(t))


def distance_from_point_to_line(point, line_point_1, line_point_2):
    x0, y0 = point
    x1, y1 = line_point_1
    x2, y2 = line_point_2
    numer = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denom = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    return numer / denom


def get_moment_of_inertia_point_point(individual_points, static_point_1, static_point_2):
    if static_point_1 == static_point_2:
        return np.nan

    r = lambda point: distance_from_point_to_line(point, static_point_1, static_point_2)
    
    # just say each point has equal mass
    return sum(r(point)**2 for point in individual_points)


def get_moment_of_inertia_point_theta(individual_points, static_point_1, theta):
    MAGNIFIER = 100
    dy = np.cos(theta) * MAGNIFIER
    dx = np.sin(theta) * MAGNIFIER
    static_point_2 = (static_point_1[0] + dx, static_point_1[1] + dy)

    return get_moment_of_inertia_point_point(individual_points, static_point_1, static_point_2)


def interpolate_nan(moments):
    for i in range(1, moments.shape[0] - 1):
        moments[i][i] = (moments[i][i - 1] + moments[i][i + 1]) / 2
    moments[0][0] = moments[0][1]
    moments[moments.shape[0] - 1][moments.shape[0] - 1] = moments[moments.shape[0] - 1][moments.shape[0] - 2]
    return moments


if __name__ == "__main__":
    RESOLUTION = 100
    xs = np.linspace(-10, 10, RESOLUTION)
    thetas = np.linspace(0, np.pi, RESOLUTION)

    f = get_parametric_function()
    f_circle = lambda t: (np.sin(t*2*np.pi/10), np.cos(t*2*np.pi/10))
    f_line = lambda t: (t + 0.01*np.sin(t), t + 0.01*np.cos(t))  # perfectly straight line results in division by zero
    f_arc = lambda t: (np.sqrt(abs(t)), t**2)

    points = f(xs)
    individual_points = list(zip(*points))

    plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot (equiv. plt.subplot(121))
    plt.plot(*points)
    plt.plot(individual_points[0][0], individual_points[0][1], "or", label="p_first")
    for i in [20, 40, 60, 80]:
        plt.plot(individual_points[i][0], individual_points[i][1], "ow")
    plt.plot(individual_points[-1][0], individual_points[-1][1], "ob", label="p_last")
    plt.title("stick")
    # plt.legend()
    # plt.show()

    moments = np.array([[np.log(get_moment_of_inertia_point_point(individual_points, p1, p2)) for p1 in individual_points] for p2 in individual_points])
    moments = interpolate_nan(moments)

    plt.subplot(1, 3, 2)
    im = plt.imshow(moments, origin="lower")
    plt.colorbar(im)
    cs = plt.contour(moments, colors="black", origin="lower")
    plt.title("moments of inertia (point, point)")

    angle_moments = np.array([[np.log(get_moment_of_inertia_point_theta(individual_points, p1, theta)) for p1 in individual_points] for theta in thetas])
    angle_moments = interpolate_nan(angle_moments)

    plt.subplot(1, 3, 3)
    im = plt.imshow(angle_moments, origin="lower")
    plt.colorbar(im)
    cs = plt.contour(angle_moments, colors="black", origin="lower")
    plt.title("moments of inertia (point, theta)")

    plt.show()
