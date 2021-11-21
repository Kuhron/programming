import random

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_parametric_function():
    norm = lambda m=0, s=1: np.random.normal(m, s)
    sin = np.sin
    cos = np.cos

    get_lin = lambda s=1: (lambda x, m=norm(s=s), b=norm(s=s): m*x + b)
    get_quad = lambda s=1: (lambda x, a=norm(s=s), b=norm(s=s), c=norm(s=s): a*x**2 + b*x + c)
    get_sin = lambda s=1: (lambda x, lin=get_lin(s=s): sin(lin(x)))
    get_simple_variable_transformation = lambda s=1: random.choice([get_lin, get_quad, get_sin])(s=s)

    def get_f_type1():
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

    def get_f_type2():
        s = 10 ** random.uniform(-2, 0)
        tr = lambda: get_simple_variable_transformation(s=s)
        n_trs = lambda n: [tr() for i in range(n)]
        t1, t2, t3, t4, t5, t6, t7, t8 = n_trs(8)
        return lambda t: t1(sin(t2(t))) + t3(sin(t4(t))**2) + t5(t)

    def get_f():
        f_type = random.choice([get_f_type1, get_f_type2, get_sin, get_lin, get_quad])
        return f_type()

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
    dy = np.sin(theta) * MAGNIFIER
    dx = np.cos(theta) * MAGNIFIER
    static_point_2 = (static_point_1[0] + dx, static_point_1[1] + dy)

    return get_moment_of_inertia_point_point(individual_points, static_point_1, static_point_2)


def interpolate_nan(moments):
    for i in range(1, moments.shape[0] - 1):
        moments[i][i] = (moments[i][i - 1] + moments[i][i + 1]) / 2
    moments[0][0] = moments[0][1]
    moments[moments.shape[0] - 1][moments.shape[0] - 1] = moments[moments.shape[0] - 1][moments.shape[0] - 2]
    return moments


def run_and_save_plots():
    RESOLUTION = 100
    ts = np.linspace(-10, 10, RESOLUTION)
    thetas = np.linspace(0, np.pi, RESOLUTION)
    desig = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    f = get_parametric_function()
    f_circle = lambda t: (np.cos(t*2*np.pi/10), np.sit(t*2*np.pi/10))
    f_line = lambda t: (t + 0.01*np.cos(t), t + 0.01*np.sin(t))  # perfectly straight line results in division by zero
    f_arc = lambda t: (np.sqrt(abs(t)), t**2)

    points = f(ts)
    individual_points = list(zip(*points))
    pxs = [p[0] for p in individual_points]
    pys = [p[1] for p in individual_points]
    xmin, xmax, ymin, ymax = min(pxs), max(pxs), min(pys), max(pys)
    x_range = xmax - xmin
    y_range = ymax - ymin
    box_width = 1.1 * max(x_range, y_range)  # force it to be a square myself
    x_avg = 0.5 * (xmin + xmax)
    y_avg = 0.5 * (ymin + ymax)
    xmin, xmax, ymin, ymax = x_avg - box_width/2, x_avg + box_width/2, y_avg - box_width/2, y_avg + box_width/2

    # plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot (equiv. plt.subplot(121))
    plt.plot(*points)
    p0x, p0y = individual_points[0]
    plt.plot(p0x, p0y, "or")
    plt.text(p0x, p0y, "  0")
    for i in [20, 40, 60, 80]:
        pix, piy = individual_points[i]
        plt.plot(pix, piy, "oy")
        plt.text(pix, piy, f"  {i}")
    pzx, pzy = individual_points[-1]
    plt.plot(pzx, pzy, "ob")
    plt.text(pzx, pzy, f"  {len(individual_points)-1}")
    plt.title("stick")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal')
    # plt.legend()
    plt.gcf().set_size_inches(6, 6)
    plt.savefig(f"MomentOfInertiaImages/{desig}_stick.png", bbox_inches="tight", pad_inches=0)
    plt.gcf().clear()

    moments = np.array([[get_moment_of_inertia_point_point(individual_points, p1, p2) for p1 in individual_points] for p2 in individual_points])
    moments_log = np.log10(moments)
    moments = interpolate_nan(moments_log)

    # plt.subplot(1, 3, 2)
    im = plt.imshow(moments_log, origin="lower")
    clb = plt.colorbar(im)
    clb.ax.set_title("log10 moment of inertia",fontsize=8)
    cs = plt.contour(moments_log, colors="black", origin="lower", linestyles="solid")
    plt.title("moments of inertia (point, point)")
    plt.gca().set_aspect('equal')
    plt.gcf().set_size_inches(6, 6)
    plt.savefig(f"MomentOfInertiaImages/{desig}_pointpoint.png", bbox_inches="tight", pad_inches=0)
    plt.gcf().clear()

    angle_moments = np.array([[get_moment_of_inertia_point_theta(individual_points, p1, theta) for p1 in individual_points] for theta in thetas])
    angle_moments_log = np.log10(angle_moments)
    angle_moments = interpolate_nan(angle_moments_log)

    # plt.subplot(1, 3, 3)
    im = plt.imshow(angle_moments_log, origin="lower")
    clb = plt.colorbar(im)
    clb.ax.set_title("log10 moment of inertia",fontsize=8)
    cs = plt.contour(angle_moments_log, colors="black", origin="lower", linestyles="solid")
    plt.title("moments of inertia (point, theta)")
    yticks = [0, 20, 40, 60, 80]
    plt.yticks(yticks)
    plt.gca().set_yticklabels([int(180/np.pi * thetas[i]) for i in yticks])  # label y axis with theta (degrees) instead of image point index
    plt.gca().set_aspect('equal')
    plt.gcf().set_size_inches(6, 6)
    plt.savefig(f"MomentOfInertiaImages/{desig}_pointangle.png", bbox_inches="tight", pad_inches=0)
    plt.gcf().clear()

    print(f"done with run {desig}")


if __name__ == "__main__":
    while True:
        run_and_save_plots()
