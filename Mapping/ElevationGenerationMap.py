import numpy as np
import matplotlib.pyplot as plt
import json
import random
import os
from datetime import datetime
import networkx as nx
from PIL import Image
from sklearn.neighbors import KDTree
from ArrayUtil import make_blank_condition_array, make_nan_array
from LatitudeLongitudeLattice import LatitudeLongitudeLattice
from IcosahedralGeodesicLattice import IcosahedralGeodesicLattice
import PlottingUtil as pu
from UnitSpherePoint import UnitSpherePoint
import ElevationChangeFunctions as elfs
import MapCoordinateMath as mcm


def add_datetime_to_fp(fp):
    split = fp.split(".")
    preceding = split[:-1]
    extension = split[-1]
    new_fp = ".".join(preceding) + "-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "." + extension
    input("check new fp is okay: {}".format(new_fp))
    return new_fp


def check_is_point_index(p):
    assert type(p) in [int, np.int64], "p = {} of type {}".format(p, type(p))


class ElevationGenerationMap:
    def __init__(self, lattice, data_dict=None, default_elevation=0):
        self.lattice = lattice
        self.data_dict = {p_i: {} for p_i in range(len(self.lattice.points))}
        if data_dict is not None:
            self.data_dict.update(data_dict)
        default_condition = lambda x: True
        self.condition_dict = {p_i: default_condition for p_i in range(len(self.lattice.points))}
        self.frozen_points = set()

    def new_value_satisfies_condition(self, p, value):
        check_is_point_index(p)
        condition = self.condition_dict.get(p)
        if callable(condition):
            res = condition(value)
            assert type(res) in [bool, np.bool_], "invalid condition return value at {} for value {}: {} of type {}".format((x, y), value, res, type(res))
            return res
        else:
            raise ValueError("invalid condition type {}".format(type(condition)))

    def fill_position(self, p, key_str, value):
        check_is_point_index(p)
        assert p not in self.frozen_points, "can't change frozen point {}".format(p)
        self.data_dict[p][key_str] = value

    def fill_point_set(self, point_set, key_str, value):
        for p in point_set:
            check_is_point_index(p)
            if p not in self.frozen_points:
                self.fill_position(p, key_str, value)

    def fill_all(self, key_str, value):
        for p in range(len(self.lattice.points)):
            check_is_point_index(p)
            self.fill_position(p, key_str, value)

    def add_value_at_position(self, p, key_str, change):
        if key_str not in self.data_dict[p]:
            self.data_dict[p][key_str] = 0
        self.data_dict[p][key_str] += change

    def get_value_at_position(self, p, key_str):
        check_is_point_index(p)
        return self.data_dict[p].get(key_str, 0)

    def get_value_array(self, key_str, points=None):
        # return all values of this str in order of point index
        if points is None:
            points = list(range(len(self.lattice.points)))
        return np.array([self.get_value_at_position(p, key_str) for p in points])

    def freeze_point(self, p):
        check_is_point_index(p)
        self.frozen_points.add(p)

    def unfreeze_point(self, p):
        check_is_point_index(p)
        self.frozen_points.remove(p)

    def unfreeze_all(self):
        self.frozen_points = set()

    def size(self):
        return self.lattice.n_points()

    def add_condition_at_position(self, p, func):
        assert callable(func)
        check_is_point_index(p)
        self.condition_dict[p] = func

    def get_neighbors(self, p):
        check_is_point_index(p)
        return self.lattice.adjacencies_by_point_index[p]

    def get_random_contiguous_region(self, p=None, radius=None, n_points=None, points_to_avoid=None, prioritize_internal_unfilled=False):
        check_is_point_index(p)
        assert int(radius is None) + int(n_points is None) == 1, "need either radius or n_points but not both, got {} and {}".format(radius, n_points)
        if points_to_avoid is None:
            points_to_avoid = set()
        points_to_avoid |= self.frozen_points
        assert all(type(x) is int for x in points_to_avoid), "points to avoid needs all indices (int) but contains other things too: {}".format(points_to_avoid)
        center_index = p
        while center_index is None or center_index in points_to_avoid:
            center_index = self.lattice.get_random_point_index()
        neighbors = [p_i for p_i in self.get_neighbors(center_index) if p_i not in points_to_avoid]
        circle = self.get_circle_around_point(center_index, radius=radius, n_points=n_points, barrier_points=points_to_avoid)
        return circle

    def get_circle_around_point(self, p, radius=None, n_points=None, barrier_points=None):
        check_is_point_index(p)
        if barrier_points is None:
            barrier_points = set()
        else:
            assert all(type(x) is int for x in barrier_points), "barrier points should be all point indices (int) but contains other types: {}".format(barrier_points)
        assert p not in barrier_points, "can't make circle with center in barrier"

        assert int(radius is None) + int(n_points is None) == 1, "need either radius or n_points but not both, got {} and {}".format(radius, n_points)

        # can tell which points are on inside vs outside of barrier wall by doing this:
        # assign weight of 0.5 to each transition into and out of the barrier (edges in the lattice's graph)
        # so each point in the lattice is associated with a number of barrier crossings
        # those ending in 0.5 are in the barrier itself, those == 1 mod 2 are on the other side
        # so take those == 0 mod 2

        g = self.lattice.graph
        xyz_as_one_sample = np.array([self.lattice.points[p].get_coords("xyz"),])
        if n_points is not None:
            subgraph_node_indices = self.lattice.kdtree.query(xyz_as_one_sample, k=n_points, return_distance=False)
            # print("queried n_points {}, got {}".format(n_points, subgraph_node_indices))
        elif radius is not None:
            # subgraph = nx.ego_graph(g, p, radius=radius)
            # print("subgraph from radius={} has {} nodes".format(radius, subgraph.number_of_nodes()))
            subgraph_node_indices = self.lattice.kdtree.query_radius(xyz_as_one_sample, radius)
            # print("queried radius {}, got {}".format(radius, subgraph_node_indices))
        else:
            raise

        assert type(subgraph_node_indices) in [list, np.ndarray], "return type was {}".format(type(subgraph_node_indices))
        assert len(subgraph_node_indices) == 1  # scikit-learn gives an array of indices for each queried point, here we only queried one
        index_array = subgraph_node_indices[0]
        assert index_array.ndim == 1  # 1D array of point indices
        subgraph_node_indices = list(index_array)
        # print("resulting subgraph node indices (len {}):".format(len(subgraph_node_indices)))
        # print(subgraph_node_indices)

        subgraph = g.subgraph(subgraph_node_indices)
        # print("subgraph from radius={}, n_points={} has {} nodes".format(radius, n_points, subgraph.number_of_nodes()))
        # print("some nodes from g: {}".format(list(g.nodes())[:5]))  # debugging what kinds of objects are in the graph

        if len(barrier_points) == 0:
            # don't do any graph computation, just return whole subgraph
            return set(subgraph.nodes)
        # add weights of 0.5 to any edge transitioning between barrier and non-barrier
        for n in subgraph.nodes:
            subgraph.nodes[n]["in_barrier"] = n in barrier_points
        for e in subgraph.edges:
            p0, p1 = e
            in_0 = p0 in barrier_points
            in_1 = p1 in barrier_points
            is_crossing = (in_0 and not in_1) or (in_1 and not in_0)
            subgraph.edges[p0, p1]["weight"] = 0.5 if is_crossing else 0

        # # debug
        # for e in subgraph.edges:
        #     p0, p1 = e
        #     print("edge {} -> {} = {}".format(p0.get_coords("latlondeg"), p1.get_coords("latlondeg"), subgraph.edges[e]["weight"]))

        # now get shortest paths from the center to every other point in the ego graph
        # the weights of these paths will indicate whether they are on the same side of the barrier or not
        points_on_same_side_of_barrier = set()
        other_points = set()  # for debugging, can look at what was selected and what wasn't
        for p1 in subgraph.nodes:
            shortest_path_length = nx.shortest_path_length(subgraph, source=p, target=p1, weight="weight")
            # print("path {} -> {} = {}".format(p.get_coords("latlondeg"), p1.get_coords("latlondeg"), shortest_path_length))
            # if weight param is a string, it will use the edge data attribute with that name
            if shortest_path_length % 2 == 0:
                points_on_same_side_of_barrier.add(p1)
            else:
                other_points.add(p1)

        # # debug: plot the subgraph and whether points were chosen or not
        # print("plotting subgraph for circle with radius {}".format(radius))
        # chosen_points_latlon = [p.get_coords("latlondeg") for p in points_on_same_side_of_barrier]
        # other_points_latlon = [p.get_coords("latlondeg") for p in other_points]
        # xs = [ll[0] for ll in chosen_points_latlon] + [ll[0] for ll in other_points_latlon]
        # ys = [ll[1] for ll in chosen_points_latlon] + [ll[1] for ll in other_points_latlon]
        # colors = ["r" for ll in chosen_points_latlon] + ["b" for ll in other_points_latlon]
        # plt.scatter(xs, ys, c=colors)
        # plt.show()

        assert p in points_on_same_side_of_barrier  # should return itself
        return points_on_same_side_of_barrier

    def get_distances_from_edge(self, point_set, use_scipy_method=True):
        # if use_scipy_method:  # TODO see if there is an equivalent "distance transform" function for arbitrary lattice points
        if False:
            pass
        #     min_x = np.inf
        #     max_x = -np.inf
        #     min_y = np.inf
        #     max_y = -np.inf
        #     for p in point_set:
        #         x, y = p
        #         min_x = min(x, min_x)
        #         max_x = max(x, max_x)
        #         min_y = min(y, min_y)
        #         max_y = max(y, max_y)
        #     
        #     # put them in an array
        #     to_relative_coords = lambda p: (p[0]-min_x, p[1]-min_y)
        #     to_absolute_coords = lambda p: (p[0]+min_x, p[1]+min_y)
        #     arr_x_size = max_x - min_x + 1
        #     arr_y_size = max_y - min_y + 1
        #     arr = np.zeros((arr_x_size, arr_y_size))
        #     rels = {}
        #     for p in point_set:
        #         xrel, yrel = to_relative_coords(p)
        #         rels[p] = (xrel, yrel)
        #         arr[xrel, yrel] = 1
        #     distance_transform_matrix = ndimage.morphology.distance_transform_edt(arr)
        #     res = {}
        #     for p in point_set:
        #         xrel, yrel = rels[p]
        #         d = distance_transform_matrix[xrel, yrel]
        #         res[p] = d - 1
        #     return res

        else:
            # old way, slower than scipy
            if len(point_set) == 0:
                return {}
            res = {}
            points_on_edge = [p for p in point_set if any(n not in point_set for n in self.get_neighbors(p))]  # flagged as slow: genexpr
            assert len(points_on_edge) > 0, "point set has no edge members:\n{}".format(sorted(point_set))
            for p in points_on_edge:
                res[p] = 0
            interior_point_set = point_set - set(points_on_edge)
            if len(interior_point_set) > 0:
                interior_distances = self.get_distances_from_edge(interior_point_set, use_scipy_method=False)
                for p, d in interior_distances.items():
                    res[p] = d + 1
            return res

    def make_random_elevation_change(self, 
            expected_change_sphere_proportion=None,
            positive_feedback_in_elevation=None,
            reference_area_ratio_at_sea_level=None,
            reference_area_ratio_at_big_abs=None,
            big_abs=None,
            critical_abs=None,  # above this abs, go farther in that direction until reach big_abs_elevation
            mu_when_small=None,
            mu_when_critical=None,  # mu for elevation change in critical zone (critical_abs <= abs(elevation) <= big_abs)
            mu_when_big=None,
            sigma_when_small=None,
            sigma_when_critical=None,
            sigma_when_big=None,
            land_proportion=None,
            spikiness=None,
        ):

        center_index = self.lattice.get_random_point_index()
        # print("making change at {}".format(center))

        if isinstance(self.lattice, IcosahedralGeodesicLattice):
            assert 0 < expected_change_sphere_proportion < 1, "expected size must be proportion of sphere surface area between 0 and 1, but got {}".format(expected_change_sphere_proportion)
            # expected size is in terms of proportion of sphere surface area; note that this does not scale linearly with radius in general
            # because will be using Euclidean distance in R^3, need to do some trig to convert the surface area proportion to 3d radius
            # center point is on the unit sphere, if r=1, what is the area within that distance? (the distance is a chord through the sphere's interior), the total sphere surface area = 4*pi*r^2 = 4*pi
            # f(r) = integral(0 to r, dA/dr dr); f(2) = the whole sphere = 4*pi; f(sqrt(2)) = half sphere = 2*pi
            # dA = 2*pi*r' dr, where r' is the radius of the flat circle that r points to, from the central axis which runs through the starting point and the sphere's center
            # drew pictures and got that r'^2 = r^2 - r^4/4; checked r'(r=sqrt(2))=1, r'(r=0)=0, r'(r=2)=0, r'(r=1)=sqrt(3)/2, all work
            # so dA = 2*pi*sqrt(r^2 - r^4/4) dr
            # but dA should be scaled up by some trig factor (e.g. it will be sqrt(2) times greater when it is slanted at 45 deg, and 0 times greater when it is vertical)
            # dA/dr' = 2*pi*dl, imagine lowering the circle by dh, so that r' rises by dr', then the slanted line on the sphere's surface is dl
            # if theta is angle with vertical axis, cos theta = dr'/dl, so dl = dr'/cos(theta), and from the center, see that sin(theta) r'/1
            # so cos(theta) = sqrt(1-r'^2), so dA = 2*pi* r' * dr'/sqrt(1-r'^2)
            # can integrate over r' instead of r now
            # f(r) = 2*pi* integral(0 to sqrt(r^2 - r^4/4), r'/sqrt(1-r'^2) dr')
            # proportion(r) = 1/(4*pi) * f(r)
            # integral(s/sqrt(1-s^2) ds) = -1*sqrt(1-s^2) => (lots of whiteboard scribbles) => proportion(r) = r^2/4 (for r in [0, 2])
            # => r(proportion) = 2*sqrt(proportion)
            radius_from_center_in_3d = 2 * np.sqrt(expected_change_sphere_proportion)
            
        else:
            raise Exception("making elevation change on non-geodesic lattice is deprecated, this lattice is a {}".format(type(self.lattice)))

        changing_reg = self.get_random_contiguous_region(center_index, radius=radius_from_center_in_3d, points_to_avoid=self.frozen_points)
        changing_reg_n_points = len(changing_reg)
        # print("proportion {} got {} changing points from lattice with {} points (got proportion {})".format(expected_change_sphere_proportion, changing_reg_n_points, len(self.lattice.points), changing_reg_n_points/len(self.lattice.points)))
        changing_reg_usps = [self.lattice.points[p_i] for p_i in changing_reg]
        ps_xyz = [np.array(p.get_coords("xyz")) for p in changing_reg_usps]
        mean_ps_xyz = sum(ps_xyz) / len(ps_xyz)  # don't use np.mean here or it will give you a single scalar, mean of all values
        assert len(mean_ps_xyz) == 3, "ya done goofed. look -> {}".format(mean_ps_xyz)
        mean_ps_xyz = np.array(mean_ps_xyz)
        mean_ps_xyz /= np.linalg.norm(mean_ps_xyz)  # normalize
        mean_ps_latlon = mcm.unit_vector_cartesian_to_lat_lon(*mean_ps_xyz, deg=True)
        coords_dict = {"xyz": tuple(mean_ps_xyz), "latlondeg": tuple(mean_ps_latlon)}
        changing_reg_center_of_mass_raw = UnitSpherePoint(coords_dict)
        changing_reg_center_of_mass = self.lattice.closest_point_to(changing_reg_center_of_mass_raw)
        com_index = self.lattice.usp_to_index[changing_reg_center_of_mass]
        e_center_of_mass = self.get_value_at_position(com_index, "elevation")
        reference_p = changing_reg_center_of_mass

        # try to get mountain chains to propagate:
        # if center point is low abs, look at bigger region, might catch mountain
        # if center point is high abs, look at smaller region, don't let lowland water it down
        if abs(e_center_of_mass) >= big_abs:
            reference_area_ratio = reference_area_ratio_at_big_abs
        else:
            slope = (reference_area_ratio_at_big_abs - reference_area_ratio_at_sea_level) / (big_abs - 0)
            reference_area_ratio = reference_area_ratio_at_sea_level + slope * abs(e_center_of_mass)

        reference_p_i = self.lattice.usp_to_index[reference_p] 
        reference_n_points = max(1, int(round(reference_area_ratio * changing_reg_n_points)))
        reference_reg = self.get_circle_around_point(reference_p_i, n_points=reference_n_points)
        elevations_in_refreg = [self.get_value_at_position(p_i, "elevation") for p_i in reference_reg]
        e_avg = np.mean(elevations_in_refreg)
        e_max = np.max(elevations_in_refreg)
        e_min = np.min(elevations_in_refreg)
        elevation_sign = (1 if e_avg > 0 else -1)

        distances = self.get_distances_from_edge(changing_reg)
        max_d = max(distances.values())
        if max_d == 0:
            raw_func = elfs.constant
        else:
            raw_func = elfs.get_elevation_change_function(spikiness=spikiness)

        if positive_feedback_in_elevation:
            # land begets land, sea begets sea
            # use e_max for detecting mountain nearby, chain should propagate
            # try to enforce land ratio approximately. change it to the other one with probability(other_sign)
            if elevation_sign == 1:
                # switch land to sea with probability(sea)
                if random.random() < (1 - land_proportion):
                    elevation_sign = -1
            elif elevation_sign == -1:
                if random.random() < (land_proportion):
                    elevation_sign = 1

            big_signed = elevation_sign * big_abs
            critical_signed = elevation_sign * critical_abs
            critical_excess = e_avg - critical_signed
            big_remainder = big_signed - e_avg
            mountain_or_trench_nearby = abs(e_max) >= big_abs or abs(e_min) >= big_abs
  
            mu = \
                elevation_sign * mu_when_big if abs(e_avg) > big_abs else \
                elevation_sign * mu_when_critical if abs(e_avg) > critical_abs else \
                elevation_sign * mu_when_small

            sigma = \
                sigma_when_big if abs(e_avg) > big_abs else \
                sigma_when_critical if abs(e_avg) > critical_abs else \
                sigma_when_small
                # can't have negative sigma!


            # if False: #mountain_or_trench_nearby:
            #     pass
            #     # try to propagate it in a line, i.e., the closer e_avg is to mountain size, the more likely it is to rise
            #     alpha_between_critical_and_big = (e_avg - critical_signed)/(big_signed - critical_signed)
            #     # closer to big = bigger alpha, bigger expected rise
            #     a = np.random.uniform(alpha_between_critical_and_big)
            #     mu = a * big_remainder
            # else:

            # try another idea, extreme elevations have expected movement of zero
            # but moderate ones move more in their direction
            # if abs(average_elevation_in_refreg) < abs(big_elevation_signed):
            #     mu = average_elevation_in_refreg
            # else:
            #     mu = 0

            # old correction to mu, I think this is causing overcorrection, might be responsible for mountain rings
            # because it sees mountains that are taller than big_abs, wants to drop them, ends up creating lowlands in the rest of the
            # changing region, could this cause a mountain ring to propagate toward shore, leaving central valley?
            # mu = average_elevation_in_refreg
            # if abs(mu) > big_abs_elevation:
            #     # decrease mu linearly down to 0 at 2*big_abs_elevation, and then drop more after that to decrease
            #     big_elevation_signed = big_abs_elevation * (1 if mu > 0 else -1)
            #     mu_excess = mu - big_elevation_signed
            #     mu -= 2*mu_excess

        else:
            mu = 0
            sigma = sigma_when_small

        # add effects of volcanism, crude approximation for now
        if abs(e_avg) < big_abs:
            # volcanism_array_of_refreg = self.get_value_array("volcanism", reference_reg)  # slow?
            volcanism_array_of_refreg = np.array([self.get_value_at_position(p_i, "volcanism") for p_i in reference_reg])
            average_volcanism_in_refreg = volcanism_array_of_refreg.mean()
            assert np.isfinite(average_volcanism_in_refreg)
            volcanism_contribution = average_volcanism_in_refreg
            # print("original mu = {}, += {} from volcanism".format(mu, volcanism_contribution))
            mu += volcanism_contribution  # positive = make mountains; negative = make rifts
            # mu = volcanism_contribution  # debug, try to get correlation to show up
        else:
            # already big, don't grow too much
            pass

        max_change = np.random.normal(mu, sigma)

        func = lambda d: raw_func(d, max_d, max_change)
        changes = {p: func(d) for p, d in distances.items() if p not in self.frozen_points}
        for p, d_el in changes.items():
            current_el = self.get_value_at_position(p, "elevation")
            new_el = current_el + d_el
            if self.new_value_satisfies_condition(p, new_el):
                self.fill_position(p, "elevation", new_el)

    # def get_random_zero_loop(self):
    #     x0_0, y0_0 = self.get_random_point(border_width=2)
    #     dx = 0
    #     dy = 0
    #     while np.sqrt(dx**2 + dy**2) < self.x_size * 1/2 * np.sqrt(2):
    #         x0_1 = random.randrange(2, self.x_size - 2)
    #         y0_1 = random.randrange(2, self.y_size - 2)
    #         dx = x0_1 - x0_0
    #         dy = y0_1 - y0_0
    #     # x0_1 = (x0_0 + self.x_size // 2) % self.x_size  # force it to be considerably far away to get more interesting result
    #     # y0_1 = (y0_0 + self.y_size // 2) % self.y_size
    #     source_0 = (x0_0, y0_0)
    #     source_1 = (x0_1, y0_1)
    #     path_0 = self.get_random_path(source_0, source_1, points_to_avoid=set())
    #     path_1 = self.get_random_path(source_0, source_1, points_to_avoid=path_0)

    #     res = path_0 | path_1
    #     # print(res)
    #     return res

    # def add_random_zero_loop(self):
    #     points = self.get_random_zero_loop()
    #     for p in points:
    #         self.fill_position(p[0], p[1], 0)

    def fill_elevations(self, 
        n_steps=None, 
        plot_every_n_steps=None,
        expected_change_sphere_proportion=None,
        positive_feedback_in_elevation=None,
        reference_area_ratio_at_sea_level=None,
        reference_area_ratio_at_big_abs=None,
        big_abs=None,
        critical_abs=None,
        mu_when_small=None,
        mu_when_critical=None,
        mu_when_big=None,
        sigma_when_small=None,
        sigma_when_critical=None,
        sigma_when_big=None,
        land_proportion=None,
        spikiness=None,
    ):
        plot_progress = type(plot_every_n_steps) is int and plot_every_n_steps > 0
        if plot_progress:
            plt.ion()
        i = 0
        t0 = datetime.now()
        while True:
            if n_steps is None:
                raise Exception("do not do this anymore, just run until there is sufficient convergence")
                # if len(self.untouched_points) == 0:
                #     break
            else:
                if i >= n_steps:
                    break
            if i % 100 == 0 or i in [1, 2, 3, 4, 5, 10, 25, 50, 75]:
                try:
                    dt = datetime.now() - t0
                    n_left = n_steps - i
                    secs_per_step = dt/i
                    eta = secs_per_step * n_left
                    eta_str = str(eta)
                    print("step {}, {} elapsed, {} ETA".format(i, dt, eta_str))
                except ZeroDivisionError:
                    pass
            self.make_random_elevation_change(
                expected_change_sphere_proportion=expected_change_sphere_proportion,
                positive_feedback_in_elevation=positive_feedback_in_elevation,
                reference_area_ratio_at_sea_level=reference_area_ratio_at_sea_level,
                reference_area_ratio_at_big_abs=reference_area_ratio_at_big_abs,
                big_abs=big_abs,
                critical_abs=critical_abs,
                mu_when_small=mu_when_small,
                mu_when_critical=mu_when_critical,
                mu_when_big=mu_when_big,
                sigma_when_small=sigma_when_small,
                sigma_when_critical=sigma_when_critical,
                sigma_when_big=sigma_when_big,
                land_proportion=land_proportion,
                spikiness=spikiness,
            )
            if plot_progress and i % plot_every_n_steps == 0:
                try:
                    self.draw()
                except ValueError:
                    print("skipping ValueError in ElevationGenerationMap.draw()")
            i += 1

    def add_fault_lines(self, 
            n_fault_tripoints=None, 
            n_volcanism_steps=None, 
            max_volcanism_change_magnitude=None, 
            min_volcanism_wavenumber=None, 
            max_volcanism_wavenumber=None,
    ):
        print("adding fault lines")
        # draw a fault line between each pair of tripoints
        tripoints = set()
        while len(tripoints) < n_fault_tripoints:
            new_tripoint = self.lattice.get_random_point_index()
            tripoints.add(new_tripoint)
        # edge_assignments = {}
        # unhappy_points = set(tripoints)  # put points here if they have 0 or 1 fault line touching them; they must have 2 or 3
        # saturated_points = set()  # put points here once they have 3 fault lines; don't accept more
        existing_fault_points = set()  # put points here so the faults won't cross
        # fault_lines_by_point = {p: 0 for p in tripoints}
        # while len(unhappy_points) > 0:
        # for a in edge_assignments:
        #     for b in edge_assignments[a]:
        #         ?
            # useable_points = tripoints - saturated_points
            # a = random.choice(list(useable_points))
            # b_candidates = list(useable_points - {a})
        for a in tripoints:
            b_candidates = list(tripoints - {a})
            xyz_a = np.array(self.lattice.points[a].get_coords("xyz"))
            b_xyzs = [np.array(self.lattice.points[bc].get_coords("xyz")) for bc in b_candidates]
            ds = [np.linalg.norm(xyz - xyz_a) for xyz in b_xyzs]
            three_neighbors = []
            for i in range(3):
                min_i = ds.index(min(ds))
                three_neighbors.append(b_candidates[min_i])
                ds.remove(ds[min_i])
            # b = b_candidates[min_index]
            # if len(useable_points) > 1:
            #     pair = random.sample(useable_points, 2)
            # except ValueError:
            #     # sample larger than population, allow a saturated point to be used
            #     assert len(useable_points) == 1, "useable points has len {}".format(len(useable_points))
            #     u_p = list(useable_points)[0]
            #     s_p = random.choice(list(tripoints))
            #     pair = (u_p, s_p)
            # a, b = pair
            for b in three_neighbors:
                other_tripoints = tripoints - {a, b}
                points_to_avoid = existing_fault_points | other_tripoints
                path = self.lattice.get_random_path(a, b, points_to_avoid)
                existing_fault_points |= path
                # fault_lines_by_point[a] += 1
                # fault_lines_by_point[b] += 1
            # for p in {a, b}:
            #     if fault_lines_by_point[p] > 1:
            #         unhappy_points -= {p}
            #     if fault_lines_by_point[p] > 2:
            #         saturated_points.add(p)
                # for p in path:
                #     self.fill_position(p, "volcanism", abs(random.normalvariate(1, 0.5)))

        # after all the lines are created
        # self.fill_point_set(existing_fault_points, "volcanism", 1)  # simplest case, use for debugging path placement
        self.fill_fault_points_with_volcanism_values(
            fault_points=existing_fault_points,
            n_volcanism_steps=n_volcanism_steps,
            max_volcanism_change_magnitude=max_volcanism_change_magnitude,
            min_volcanism_wavenumber=min_volcanism_wavenumber,
            max_volcanism_wavenumber=max_volcanism_wavenumber,
        )
        print("there are {} fault points out of {} total".format(len(existing_fault_points), len(self.lattice.points)))
        print("- done adding fault lines")

    def fill_fault_points_with_volcanism_values(self, 
            fault_points=None, 
            n_volcanism_steps=None, 
            max_volcanism_change_magnitude=None,
            min_volcanism_wavenumber=None,  # number of crests of volcanism change sin wave over the planet; wavenumber 1 is a sigmoid
            max_volcanism_wavenumber=None,
        ):
        # positive volcanism means convergent boundary / flow of rock outward onto the surface (volcano)
        # negative volcanism means divergent boundary / sinking of rock into the interior (trench, rift)
        # make the values vary like random waves around the whole subgraph

        # TODO might be nice to implement a more general "create data" function that uses the elevation logic
        # - and apply that to any subgraph/lattice, so here can just pass it only the fault points instead of the whole globe
        ps = list(fault_points)  # order them in case they're not
        sg = self.lattice.graph.subgraph(ps)
        # print("cycles:", nx.cycle_basis(sg))
        xyz_coords = np.array([self.lattice.points[p].get_coords("xyz") for p in ps])
        sg_kdtree = KDTree(xyz_coords)
        n_steps = n_volcanism_steps  # more steps adds more noise, makes the individual waves less obvious, so it looks more natural
        for step_i in range(n_steps):
            i = random.randrange(len(ps))
            p = ps[i]
            xyz_p = xyz_coords[i]
            print("chose point i={} p={} with coords {}".format(i, p, xyz_p))
            # apply some sin wave to every point on the map based on distance from p
            query = np.array([xyz_p])
            distances, neighbors = sg_kdtree.query(query, k=len(ps))
            assert distances.shape[0] == 1  # one sub-array corresponding to the single query point
            assert neighbors.shape[0] == 1  # one sub-array corresponding to the single query point
            distances = distances[0]
            neighbors_raw = neighbors[0]
            # note that neighbors here is the indices of the points IN THE KDTREE, NOT THEIR POINT INDEX IN THE LATTICE
            # - this was the source of the Sierpinski error (it reflected how I numbered the points in icosahedron construction)
            neighbors = [ps[n_i] for n_i in neighbors_raw]
            # maximum distance can be 2 (2 radii away along diameter of unit sphere)
            assert max(distances) <= 2
            assert all(n in ps for n in neighbors), "got neighbors outside of point set"
            # ideally function is smooth
            # set p to some value, set antipodes to zero
            # sigmoid_01 = lambda x, b=-2: 1/(1+(x/(1-x))**-b)  # https://stats.stackexchange.com/questions/214877/
            sin_wave = lambda x, k: (1+np.sin(((-1)**k)*(2*k+1)*np.pi*(x+0.5)))/2  # for k >= 0, this gives sin wave from [0,1] to [0,1] with f(0)=1, f(1)=0, and k+1 crests
            volcanism_sign = [-1, 1][step_i % 2]  # alternate so you don't get too much bias in either direction
            max_change = random.uniform(0, max_volcanism_change_magnitude) * volcanism_sign
            get_k = lambda: random.randint(min_volcanism_wavenumber, max_volcanism_wavenumber)
            f = lambda d, y=max_change: sin_wave(d/2, k=get_k()) * y
            # f = lambda d: 10 + random.random() * 5 + 0*d  # debug
            changes = f(distances)
            for neighbor, change in zip(neighbors, changes):
                self.add_value_at_position(neighbor, "volcanism", change)

        # now even it out to total volcanism of zero
        volcanism_array = self.get_value_array("volcanism")
        total_volcanism = sum(volcanism_array)
        # only apply the adjustment to fault-line points
        n_fault_points = len(ps)
        print("total volcanism is {} over {} points".format(total_volcanism, n_fault_points))
        adjustment_per_point = -1 * total_volcanism / n_fault_points
        for p in ps:
            self.add_value_at_position(p, "volcanism", adjustment_per_point)
        post_adjustment_total_volcanism = sum(self.get_value_array("volcanism"))
        if abs(post_adjustment_total_volcanism) > 1e-6:
            raise Exception("non-zero total volcanism persists: {}".format(post_adjustment_total_volcanism))
        
    def add_hotspots(self, n_hotspots, hotspot_min_magnitude_factor, hotspot_max_magnitude_factor):
        max_abs_volcanism = max(abs(self.get_value_array("volcanism")))
        hotspot_min_val = hotspot_min_magnitude_factor * max_abs_volcanism  # note this is still multiplied by MAX of other volcanism magnitude
        hotspot_max_val = hotspot_max_magnitude_factor * max_abs_volcanism
        for i in range(n_hotspots):
            p = self.lattice.get_random_point_index()
            self.add_value_at_position(p, "volcanism", random.uniform(hotspot_min_val, hotspot_max_val))

    def plot(self):
        # plt.gcf()
        self.pre_plot()
        plt.show()

    def draw(self):
        plt.gcf().clear()
        self.pre_plot()
        plt.draw()
        plt.pause(0.001)

    def save_plot_image(self, key_str, project_name, project_version, size_inches=None, cmap=None):
        # output_fp = add_datetime_to_fp(output_fp)
        output_fp = "/home/wesley/programming/Mapping/Projects/{project_name}/Plots/EGP_{project_name}_{key_str}_v{project_version}.png".format(**locals())
        while os.path.exists(output_fp):
            print("file {} exists, renaming output fp".format(output_fp))
            output_fp = output_fp.replace(".png", "-1.png")
        print("saving plot image to {}".format(output_fp))
        self.pre_plot(key_str, size_inches=size_inches, cmap=cmap)
        plt.savefig(output_fp)
        print("- done saving plot image")

    def pre_plot(self, key_str, size_inches=None, cmap=None):
        self.lattice.plot_data(self.data_dict, key_str, size_inches=size_inches, cmap=cmap)

    def plot_gradient(self):
        ax1 = plt.subplot(1, 2, 1)
        self.create_gradient_direction_plot()
        plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        self.create_gradient_magnitude_plot()
        plt.show()

    def plot_gradient_magnitude(self):
        self.create_gradient_magnitude_plot()
        plt.show()

    def plot_map_and_gradient_magnitude(self):
        ax1 = plt.subplot(1, 2, 1)
        self.pre_plot()
        plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        self.create_gradient_magnitude_plot()
        plt.show()

    def create_gradient_direction_plot(self):
        plt.title("gradient direction")
        varray = self.array
        vgrad = np.gradient(varray)
        grad_angle = np.angle(vgrad[0] + 1j*vgrad[1])
        angle_colormap = plt.cm.hsv  # something cyclic
        angle_color = angle_colormap(grad_angle)
        plt.imshow(grad_angle, cmap=angle_colormap, vmin=-np.pi, vmax=np.pi)
        plt.colorbar()

    def create_gradient_magnitude_plot(self):
        plt.title("gradient magnitude")
        varray = self.array
        vgrad = np.gradient(varray)
        grad_mag = np.sqrt(vgrad[0]**2 + vgrad[1]**2)
        mag_colormap = plt.cm.gist_rainbow  # most gradients are near zero, want even slightly higher ones to stand out
        plt.imshow(grad_mag, cmap=mag_colormap)
        plt.colorbar()

    def plot_volcanism_data(self):
        cmap = pu.get_volcanism_colormap()
        self.lattice.plot_data(self.data_dict, "volcanism", cmap=cmap)
        plt.show()

    def create_rainfall_array(self):
        if hasattr(self, "rainfall_array") and self.rainfall_array is not None:
            return
        self.rainfall_array = np.random.uniform(0, 1, size=self.array.shape)
        water_points = self.array < 0  # is_land changes means this changes
        self.rainfall_array[water_points] = 0
        # could have negative values correspond to more evaporation than rain
        # treat units as height units per tick of time, for flow simulation

    def create_flow_arrays(self):
        # div(water_flow) is zero everywhere, whether it leaves by flowing or evaporating or whatever
        # so water flow array should tell what the volumetric flow *through* the point is
        self.create_rainfall_array()
        flow_array = np.zeros((self.x_size, self.y_size))
        flow_destination_array = np.full((self.x_size, self.y_size), None)
        # treat sea level as fixed, water flow into and out of elevations below 0 is ignored
        points_sorted_by_decreasing_elevation = []
        for x in self.x_range:
            for y in self.y_range:
                if self.is_land(x, y):
                    elevation = self.array[x, y]
                    tup = (elevation, x, y)
                    points_sorted_by_decreasing_elevation.append(tup)
        points_sorted_by_decreasing_elevation = sorted(points_sorted_by_decreasing_elevation, reverse=True)
        gx, gy = np.gradient(-1*self.array)
        downhill_neighbor_offset = {
            -180: (-1, 0),
            # -135: (-1, -1),
            -90:  (0, -1),
            # -45:  (1, -1),
            0:    (1, 0),
            # 45:   (1, 1),
            90:   (0, 1),
            # 135:  (-1, 1),
            180:  (-1, 0),
        }
        # 8 neighbors allows rivers to cross each other diagonally, so use 4
        for el, x, y in points_sorted_by_decreasing_elevation:
            dx = gx[x, y]
            dy = gy[x, y]
            grad_angle = np.angle(dx + 1j*dy, deg=True)
            # print("dx {} dy {} angle {} deg".format(dx, dy, grad_angle))
            # rounded_to_45_deg = int(45*round(grad_angle/45))
            rounded_to_90_deg = int(90*round(grad_angle/90))
            # input("rounded to {}".format(rounded_to_45_deg))
            downhill_x_offset, downhill_y_offset = downhill_neighbor_offset[rounded_to_90_deg]
            downhill_neighbor = (x + downhill_x_offset, y + downhill_y_offset)
            nx, ny = downhill_neighbor
            flow_array[x, y] += self.rainfall_array[x, y]
            if self.is_valid_point(nx, ny):
                flow_array[nx, ny] += flow_array[x, y]
                flow_destination_array[x, y] = (nx, ny)
            # else:
            #     input("invalid neighbor {}".format((nx, ny)))
        self.flow_array = flow_array
        self.flow_destination_array = flow_destination_array

        qs = np.linspace(0, 1, 100)
        flow_array_no_zeros = flow_array[flow_array != 0]
        quantiles = {q: np.quantile(flow_array_no_zeros, q) for q in qs}
        def get_nearest_quantile(x):
            if x < quantiles[qs[0]]:
                return 0
            if x > quantiles[qs[-1]]:
                return 1
            for q0, q1 in zip(qs[:-1], qs[1:]):
                v0 = quantiles[q0]
                v1 = quantiles[q1]
                # print(v0, x, v1)
                if v0 <= x <= v1:
                    # input("match")
                    if abs(q0-x) > abs(q1-x):
                        return q0
                    else:
                        return q1
            raise RuntimeError("no quantile found for value {}".format(x))

        flow_quantile_array = np.zeros((self.x_size, self.y_size))
        for x in self.x_range:
            for y in self.y_range:
                flow_quantile_array[x, y] = get_nearest_quantile(flow_array[x, y])
        self.flow_quantile_array = flow_quantile_array

        self.water_depth_array = np.zeros((self.x_size, self.y_size))
        water_points = self.array < 0  # is_land change means this changes
        self.water_depth_array[water_points] = -1*self.array[water_points]
        
    def is_land(self, x, y):
        # TODO: make it possible for land to be below sea level
        return self.array[x, y] >= 0

    def apply_rainfall(self):
        self.water_depth_array += self.rainfall_array
        total_height_array = self.array + self.water_depth_array
        depth_changes_array = np.zeros((self.x_size, self.y_size))
        for x in self.x_range:
            for y in self.y_range:
                if not self.is_land(x, y):
                    # don't transfer from sea to anywhere else
                    continue
                h_this_point = total_height_array[x, y]
                ns = self.get_neighbors(x, y, mode=4)
                neighbors_leq_total = [n for n in ns if total_height_array[n[0], n[1]] <= h_this_point]
                if len(neighbors_leq_total) == 0:
                    continue
                n_heights = [total_height_array[n[0], n[1]] for n in neighbors_leq_total]
                heights_and_neighbors_increasing = sorted(zip(n_heights, neighbors_leq_total))
                # print("\n{} h={}\nneighbors sorted {}".format((x, y), h_this_point, heights_and_neighbors_increasing))
                # now distribute the height difference to the neighbors such that:
                # lowest heights first, add to all the heights currently ranked lowest until they tie with the next rank
                # then add to all those equally, etc. until they are equal to the remaining height of this point
                while True:
                    lowest_h = heights_and_neighbors_increasing[0][0]
                    lowest_to_this_point_dh = h_this_point - lowest_h
                    current_water_depth_this_point = self.water_depth_array[x, y] + depth_changes_array[x, y]
                    max_amount_can_transfer = current_water_depth_this_point  # times 1 for the area of the spot it is on
                    max_amount_can_transfer /= 4  # viscosity?? will hopefully prevent checkerboard alternation
                    next_h = None
                    lowest_rank = [heights_and_neighbors_increasing[0]]
                    for nh, n in heights_and_neighbors_increasing[1:]:
                        if nh > lowest_h:
                            next_h = nh
                            break  # for
                        lowest_rank.append((nh, n))
                    if next_h is None:
                        # everything in neighbors was same height
                        next_h = h_this_point

                    # all points that are tied for lowest height will get an equal share of the flow
                    n_receivers = len(lowest_rank)
                    # first, get how much would be transferred to them to equalize with this point
                    average_h = 1/(n_receivers+1) * (n_receivers*lowest_h + 1*h_this_point)
                    lowest_to_average_dh = average_h - lowest_h
                    # but if this is more than they would take to go to the next highest neighbor,
                    # then they will equilibrate with it so we loop again
                    lowest_to_next_dh = next_h - lowest_h
                    dh_to_implement = min(lowest_to_average_dh, lowest_to_next_dh)
                    assert dh_to_implement >= 0

                    amount_to_transfer = dh_to_implement * n_receivers
                    if amount_to_transfer > max_amount_can_transfer:
                        amount_to_transfer = max_amount_can_transfer
                        will_break = True
                    else:
                        will_break = False

                    amount_to_transfer_to_each = amount_to_transfer / n_receivers
                    # print("lowest_rank {} with {} receivers".format(lowest_rank, n_receivers))
                    
                    if amount_to_transfer == 0:
                        break
                    for i in range(n_receivers):
                        n = lowest_rank[i][1]
                        if not self.is_land(*n):
                            # sea will pull higher water level toward it, but then acts like an infinite sink, sea level will not rise
                            continue
                        depth_changes_array[n[0], n[1]] += amount_to_transfer_to_each
                        heights_and_neighbors_increasing[i] = (
                            heights_and_neighbors_increasing[i][0] + amount_to_transfer_to_each,
                            heights_and_neighbors_increasing[i][1]
                        )
                    depth_changes_array[x, y] -= amount_to_transfer  # total transferred
                    resulting_depth = self.water_depth_array[x, y] + depth_changes_array[x, y]
                    # print("depth change at {} -= {} --> {}\nwill give depth {}".format((x, y), amount_to_transfer, depth_changes_array[x, y], resulting_depth))
                    if resulting_depth < 0:
                        if abs(resulting_depth) > 1e-6:
                            print("uh oh, negative depth created")
                            input("check")
                    if will_break:
                        break  # while
                # next point, don't apply depth changes until very end so you do all points at once based on what they wanted to do at this time
        # print("\ngot depth changes array with sum {}, max abs {}:\n{}".format(depth_changes_array.sum(), abs(depth_changes_array).max(), depth_changes_array))
        self.water_depth_array += depth_changes_array
        # print("resulting water depth array:\n{}".format(self.water_depth_array))
        # input("check")
        assert self.water_depth_array.min() >= -1e-6, "no negative water depth allowed"

    def get_average_water_depth(self, initial_steps, averaging_steps):
        for i in range(initial_steps):
            print("initialization step", i)
            # don't average over these, let it try to reach a stable state
            self.apply_rainfall()
        sum_water_depth_array = np.zeros((self.x_size, self.y_size))
        for i in range(averaging_steps):
            print("averaging step", i)
            self.apply_rainfall()
            sum_water_depth_array += self.water_depth_array
        return sum_water_depth_array / averaging_steps

    def plot_flow_steps(self, n_steps):
        plt.ion()
        for _ in range(n_steps):
            self.apply_rainfall()
            plt.gcf().clear()
            plt.imshow(self.array + self.water_depth_array)
            plt.colorbar()
            plt.draw()
            plt.pause(0.001)

    def plot_average_water_location(self):
        average_water_depth_array = self.get_average_water_depth(100, 100)
        average_water_depth_array[self.array < 0] = 0  # is_land changes means this changes
        average_height_array = self.array + average_water_depth_array
        plt.subplot(1, 2, 1)
        plt.imshow(average_water_depth_array)
        plt.subplot(1, 2, 2)
        plt.imshow(average_height_array)
        plt.colorbar()
        plt.show()

    def plot_flow_amounts(self):
        self.create_flow_arrays()
        # arr = self.flow_array  # max is too high for linear cmap
        arr = self.flow_quantile_array
        plt.imshow(arr, cmap=plt.cm.inferno)
        plt.colorbar()
        plt.show()

    def plot_rivers(self):
        self.pre_plot(alpha=1)
        self.create_flow_arrays()
        print("flow stats: min {} median {} mean {} max {}".format(
            np.min(self.flow_array),
            np.median(self.flow_array),
            np.mean(self.flow_array),
            np.max(self.flow_array),
        ))
        mean_flow = np.mean(self.flow_array)

        # blue_value = lambda x: get_nearest_quantile(x)
        # alpha_value = lambda x: get_nearest_quantile(x)
        # river_rgba_array = []
        line_segments = []
        colors = []
        for x in self.x_range:
            # this_row = []
            for y in self.y_range:
                flow = self.flow_array[x, y]
                flow_destination = self.flow_destination_array[x, y]
                flow_quantile = self.flow_quantile_array[x, y]
                if flow_destination is not None:
                    # seg = [(x, y), flow_destination]
                    # transpose
                    seg = [(y, x), flow_destination[::-1]]
                    # print(seg)
                    line_segments.append(seg)
                    # b = blue_value(flow)
                    # a = alpha_value(flow)
                    # r = 0
                    # g = b  # make it cyan
                    # color = (r, g, b, a)
                    # color = (0, 0, 1, a)
                    # if a > 0:
                    #     input("flow {} from {} to {} gave rgba {}".format(flow, (x, y), flow_destination, color))
                    cmap = plt.cm.GnBu
                    r, g, b, a = cmap(flow_quantile)
                    color = (r, g, b, a*0.5)
                    colors.append(color)
                # this_row.append(color)
            # river_rgba_array.append(this_row)
        # river_rgba_array = np.array(river_rgba_array)
        # print(test_array)
        # print(f(test_array))
        # print(river_rgba_array[150:152, 150:153])
        # plt.imshow(self.water_flow_array)
        # plt.imshow(river_rgba_array, origin="lower")
        lc = mcollections.LineCollection(line_segments, colors=colors)
        plt.gca().add_collection(lc)
        plt.gca().autoscale()
        plt.show()

    @staticmethod
    def from_image(image_fp, color_condition_dict, default_color, latlon00, latlon01, latlon10, latlon11, map_lattice):
        print("called ElevationGenerationMap.from_image()")
        # all points in image matching something in the color dict should be that color no matter what
        # everything else is randomly generated
        # i.e., put the determined points in points_to_avoid for functions that take it

        if any(len(x) != 4 for x in color_condition_dict.keys()):
            raise ValueError("all color keys must have length 4, RGBA:\n{}".format(color_condition_dict.keys()))

        im = Image.open(image_fp)
        width, height = im.size
        print("creating image lattice")
        image_lattice = LatitudeLongitudeLattice(
            height, width,  # rows, columns
            latlon00, latlon01, latlon10, latlon11
        )  # we are not actually going to add data to this lattice, but we will use it to get point coordinates more easily
        # later will need to be able to adjust edge length dynamically based on how fine the image grid is
        print("- done creating image lattice")
        m = ElevationGenerationMap(map_lattice)
        arr = np.array(im)
        color_and_first_seen = {}
        usp_to_fill_value = {}
        usp_to_condition = {}
        print("mapping image points to value and condition")
        for x in range(height):
            if x % 100 == 0:
                print("x = {}/{}".format(x, height))
            for y in range(width):
                color = tuple(arr[x, y])
                if color not in color_condition_dict:
                    color = default_color
                    arr[x, y] = color
                if color not in color_and_first_seen:
                    print("got new color {} at {}".format(color, (x, y)))
                    color_and_first_seen[color] = (x, y)
                if color in color_condition_dict:
                    fill_value, condition, is_frozen = color_condition_dict[color]
                    if fill_value is None:
                        fill_value = 0
                    xy = (x, y)
                    p = image_lattice.get_usp_from_lattice_position(xy)
                    usp_to_fill_value[p] = fill_value
                    usp_to_condition[p] = condition
        print("- done mapping points to value and condition")

        print("creating map from lattice points to image points")
        map_lattice_points = map_lattice.points
        print("creating image points that will be referenced")
        image_points_that_will_be_referenced = set()
        for i, lattice_p in enumerate(map_lattice_points):
            if i % 100 == 0:
                # can be optimized by not checking map lattice points that are too far away from the image edges, reduce number of distance calculations that will just be wasted
                print("map point {}/{}".format(i, len(map_lattice_points)))
            image_point = image_lattice.closest_point_to(lattice_p)
            image_points_that_will_be_referenced.add(image_point)
        print("- done creating image points that will be referenced")
        # now for each of these image points, get the closest lattice point and use the image point for that lattice point, not others
        # because the others claiming the same closest image point will be outside the image boundaries, I think
        lattice_points_to_image_points = {}
        for i, image_p in enumerate(image_points_that_will_be_referenced):
            if i % 100 == 0:
                print("image point {}/{}".format(i, len(image_points_that_will_be_referenced)))
            key = map_lattice.closest_point_to(image_p)
            val = image_p
            lattice_points_to_image_points[key] = val
        print("- done creating map from lattice points to image points")

        print("filling map values and conditions")
        for lattice_p, image_p in lattice_points_to_image_points.items():
            # faster to get the image point near each lattice point than the other way around, if the lattice is lower-resolution
            # if the image is lower-resolution than the lattice, then start doing things the other way around, maybe?
                # image_point = image_lattice.point_dict[(x, y)] 
                # p = map_lattice.closest_point_to(image_point)  # closest point on lattice to this x, y point on the image
            fill_value = usp_to_fill_value[image_p]
            condition = usp_to_condition[image_p]
            m.fill_position(lattice_p, "elevation", fill_value)
            m.add_condition_at_position(lattice_p, condition)
            if is_frozen:
                m.freeze_point(lattice_p)
        print("- done filling map values and conditions")

        print("- returning ElevationGenerationMap from image")
        return m

    @staticmethod
    def from_data(key_strs, project_name, project_version):
        assert type(key_strs) is list, "invalid key_strs: {}".format(key_strs)
        print("loading data {} for project {} v{}".format(key_strs, project_name, project_version))
        data_dict = {}
        for key_str in key_strs:
            data_dict_this_key = ElevationGenerationMap.load_single_data_file(key_str, project_name, project_version)
            for p_i in data_dict_this_key:
                if p_i not in data_dict:
                    data_dict[p_i] = {}
                data_dict[p_i][key_str] = data_dict_this_key[p_i][key_str]
        n_iterations = IcosahedralGeodesicLattice.get_iterations_from_number_of_points(len(data_dict))
        lattice = IcosahedralGeodesicLattice(iterations=n_iterations)

        return ElevationGenerationMap(lattice=lattice, data_dict=data_dict)

    @staticmethod
    def load_single_data_file(key_str, project_name, project_version):
        assert type(key_str) is str, "invalid key_str: {}".format(key_str)
        data_fp = "/home/wesley/programming/Mapping/Projects/{project_name}/Data/EGD_{project_name}_{key_str}_v{project_version}.txt".format(**locals())
        with open(data_fp) as f:
            contents = f.read()
        vals = contents.split("\n")
        if vals[-1] == "":
            vals = vals[:-1]
        vals = [float(x) for x in vals]
        data_dict = {}  # point index to value
        for p_i, val in enumerate(vals):
            data_dict[p_i] = {key_str: val}
        # print("got data_dict from data_fp {}: {}".format(data_fp, data_dict))
        # input("check for correctness")
        return data_dict

        # older, for LatitudeLongitudeLattice
        # array = np.array(lines)
        # x_size, y_size = array.shape
        # return ElevationGenerationMap(x_size, y_size, latlon00, latlon01, latlon10, latlon11, array=array)

    def freeze_coastlines(self):
        coastal_points = set()
        for p in range(len(self.lattice.points)):
            if self.get_value_at_position(p, "elevation") < 0:
                neighbors = self.get_neighbors(p)
                for n in neighbors:
                    if self.get_value_at_position(n, "elevation") >= 0:
                        coastal_points.add(n)
        for p in coastal_points:
            self.fill_position(p, "elevation", 0)
            self.freeze_point(p)

    # def get_min_gradient_array(self):
    #     if hasattr(self, "min_gradient_array") and self.min_gradient_array is not None:
    #         return self.min_gradient_array
    #     res = make_nan_array(self.x_size, self.y_size)
    #     for p in self.get_all_points():
    #         min_grad_this_point = np.inf
    #         for q in self.get_neighbors(*p):
    #             dist = 1 if p[0] == q[0] or p[1] == q[1] else np.sqrt(2)
    #             dh = self.array[q[0], q[1]] - self.array[p[0], p[1]]
    #             grad = dh/dist
    #             min_grad_this_point = min(grad, min_grad_this_point)
    #         res[p[0], p[1]] = min_grad_this_point
    #     self.min_gradient_array = res
    #     return res

    def get_max_gradient(self):
        print("getting max grad")
        from_point = None
        to_point = None
        max_grad = -np.inf
        max_grad_pair = None
        all_points = sorted(self.get_all_points())
        for p in all_points:
            for q in self.get_neighbors(*p):
                dist = 1 if p[0] == q[0] or p[1] == q[1] else np.sqrt(2)
                dh = self.array[q[0], q[1]] - self.array[p[0], p[1]]
                grad = dh/dist
                if grad > max_grad:
                    max_grad = grad
                    max_grad_pair = (p, q)
        return max_grad, max_grad_pair

    def save_data(self, key_str, project_name, project_version):
        # format is just grid of comma-separated numbers
        # if not confirm_overwrite_file:
        #     return
        # output_fp = add_datetime_to_fp(output_fp)
        output_fp = "/home/wesley/programming/Mapping/Projects/{project_name}/Data/EGD_{project_name}_{key_str}_v{project_version}.txt".format(**locals())
        while os.path.exists(output_fp):
            print("file {} exists, renaming output fp".format(output_fp))
            output_fp = output_fp.replace(".txt", "-1.txt")
        print("saving {} data to {}".format(key_str, output_fp))
        with open(output_fp, "w") as f:
            s = ""
            for p_i in range(len(self.lattice.points)):
                val = self.get_value_at_position(p_i, key_str)
                s += str(val) + "\n"
            f.write(s)
        print("finished saving {} data".format(key_str))
