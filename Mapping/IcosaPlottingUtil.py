import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
import scipy
from scipy import interpolate  # it seems on WSL I have to do this, importing the submodule, while on my Ubuntu laptop this is not necessary


import IcosahedronMath as icm


def scatter_icosa_points_by_number(point_numbers, show=True):
    point_codes = icm.get_point_codes_from_point_numbers(point_numbers)
    scatter_icosa_points_by_code(point_codes, show=show)


def scatter_icosa_points_by_code(point_codes, show=True, **kwargs):
    latlons = icm.get_latlons_from_point_codes(point_codes)
    lats = [ll[0] for ll in latlons]
    lons = [ll[1] for ll in latlons]
    plt.scatter(lons, lats, **kwargs)
    if show:
        plt.show()


def plot_neighbor_relationships(n_iterations):
    d = icm.get_adjacency_memo_dict(n_iterations)
    n_points = icm.get_exact_n_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    neighbor_indices = range(6)
    colors = ["red","yellow","green","blue","purple","black"]
    for ni, c in zip(neighbor_indices, colors):
        neighbor_numbers_at_index = [d[pi][ni] for pi in point_numbers]
        plt.scatter(point_numbers, neighbor_numbers_at_index, color=c, alpha=0.4)
    plt.show()


def plot_xyzs(n_iterations):
    d = icm.get_position_memo_dict(n_iterations)
    n_points = icm.get_exact_n_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    xs = [d[pi]["xyz"][0] for pi in point_numbers]
    ys = [d[pi]["xyz"][1] for pi in point_numbers]
    zs = [d[pi]["xyz"][2] for pi in point_numbers]

    plt.scatter(point_numbers, xs)
    plt.title("x")
    plt.show()

    plt.scatter(point_numbers, ys)
    plt.title("y")
    plt.show()

    plt.scatter(point_numbers, zs)
    plt.title("z")
    plt.show()


def plot_latlons(n_iterations):
    d = icm.get_position_memo_dict(n_iterations)
    n_points = icm.get_exact_n_points_from_iterations(n_iterations)
    point_numbers = range(12, n_points)
    lats = [d[pi]["latlondeg"][0] for pi in point_numbers]
    lons = [d[pi]["latlondeg"][1] for pi in point_numbers]
    
    plt.scatter(point_numbers, lats)
    plt.title("lat")
    plt.show()

    plt.scatter(point_numbers, lons)
    plt.title("lon")
    plt.show()


def plot_coordinate_patterns(n_iterations):
    # for trying to get some pattern recognition and figure out what the functions are that determine the positions and adjacencies of the icosahedron points
    # right now it seems pretty hopeless; there are a lot of complicated patterns, they look cool but I don't understand them
    # the X plot has Sierpinski fractals, lots of other fractal structures visible at high iteration numbers (~7)
    plot_neighbor_relationships(n_iterations)
    plot_xyzs(n_iterations)
    plot_latlons(n_iterations)


def plot_variable_at_point_codes(pcs, db, variable_name, xyzg, show=True):
    df = db.df
    df2 = df.loc[pcs,:]
    latlons = icm.get_latlons_from_point_codes(pcs, xyzg)
    lats = [ll[0] for ll in lls]
    lons = [ll[1] for ll in lls]
    variable_values = df2.loc[:, variable_name]
    plt.scatter(lons, lats, c=variable_values)
    plt.colorbar()
    plt.title(variable_name)
    if show:
        plt.show()


def plot_variable_scattered_from_db(db, pcs, var_to_plot, show=True):
    print(f"plotting variable scattered: {var_to_plot}")
    pc_to_val = db.get_dict(pcs, var_to_plot)
    plot_variable_scattered_from_dict(pc_to_val, title=var_to_plot, show=show)


def plot_variable_scattered_from_dict(pc_to_val, xyzg, title=None, show=True):
    pcs = list(pc_to_val.keys())
    latlons = icm.get_latlons_from_point_codes(pcs, xyzg)
    lats = [latlon[0] for latlon in latlons]
    lons = [latlon[1] for latlon in latlons]
    vals = [pc_to_val.get(pc) for pc in pcs]
    plt.scatter(lons, lats, c=vals)
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()


def plot_variables_scattered_from_db(db, pcs, vars_to_plot):
    print(f"plotting variables scattered: {vars_to_plot}")
    n_plots = len(vars_to_plot)
    for i, var in enumerate(vars_to_plot):
        plt.subplot(1, n_plots, i+1)
        plot_variable_scattered_from_db(db, pcs, var, show=False)
    plt.show()


def plot_variable_interpolated_from_db(db, lns, var_to_plot, xyzg, resolution, show=True):
    print(f"plotting variable interpolated: {var_to_plot}")
    ln_to_val = db.get_dict(lns, var_to_plot)
    plot_variable_interpolated_from_dict(ln_to_val, xyzg, resolution, title=None, show=show)


def plot_variable_interpolated_from_dict(ln_to_val, xyzg, resolution, title=None, show=True):
    lns = list(ln_to_val.keys())
    pcs = icm.get_point_codes_from_prefix_lookup_numbers(lns)
    latlons = icm.get_latlons_from_point_codes(pcs, xyzg)
    values = [ln_to_val.get(ln) for ln in lns]
    plot_interpolated_data(latlons, values, lat_range=None, lon_range=None, n_lats=resolution, n_lons=resolution)
    if show:
        plt.show()


def plot_variables_interpolated_from_db(db, lns, vars_to_plot, xyzg, resolution, show=False):
    print(f"plotting variables interpolated: {vars_to_plot}")
    n_plots = len(vars_to_plot)
    for i, var in enumerate(vars_to_plot):
        plt.subplot(1, n_plots, i+1)
        plot_variable_interpolated_from_db(db, lns, var, xyzg, resolution, show=False)
        plt.title(var)
    if show:
        plt.show()


def plot_variable_world_map_from_db(db, var_to_plot, xyzg, pixels_per_degree, show=False):
    lns = db.df.index
    pcs = icm.get_point_codes_from_prefix_lookup_numbers(lns)
    # pcs = random.sample(list(pcs), 10000)  # debug
    print("getting latlons")
    latlons = icm.get_latlons_from_point_codes(pcs, xyzg)
    print("getting values")
    values = db.df.loc[lns, var_to_plot]
    n_lats = int(2*90*pixels_per_degree)
    n_lons = int(2*180*pixels_per_degree)
    print("plotting interpolated")
    plot_interpolated_data(latlons, values, lat_range=(-90, 90), lon_range=(-180, 180), n_lats=n_lats, n_lons=n_lons)
    if show:
        plt.show()


def plot_latlons(pcs, xyzg):
    latlons = icm.get_latlons_from_point_codes(pcs, xyzg)
    lats = [latlon[0] for latlon in latlons]
    lons = [latlon[1] for latlon in latlons]
    plt.scatter(lons, lats)
    plt.show()
