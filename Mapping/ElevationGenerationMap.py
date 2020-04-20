import numpy as np
from PIL import Image
from ArrayUtil import make_blank_condition_array, make_nan_array


class ElevationGenerationMap:
    def __init__(self, lattice, array=None):
        # really row size and column size, respectively
        self.lattice = lattice
        # if array is None:
        #     self.array = make_nan_array((x_size, y_size))
        # else:
        #     self.array = array
        # self.condition_array = make_blank_condition_array((x_size, y_size))
        self.frozen_points = set()
        # self.neighbors_memoized = {}
        # self.memoize_all_neighbors()

    # def memoize_all_neighbors(self):  # now should be handled by the lattice
    #     # just calling get_neighbors will memoize
    #     # corner pixels
    #     for x, y in (0, 0), (0, self.y_size-1), (self.x_size-1, 0), (self.x_size, self.y_size):
    #         n = self.get_neighbors(x, y)
    #     # edge representatives
    #     for x, y in (0, 1), (1, 0), (1, self.y_size-1), (self.x_size-1, 1):
    #         n = self.get_neighbors(x, y)
    #     # interior representative
    #     n = self.get_neighbors(1, 1)

    #     # old way, memory hog
    #     # for x in range(self.x_size):
    #     #     for y in range(self.y_size):
    #     #         n = self.get_neighbors(x, y)
    #     #         # function will memoize them
    #     # assert len(self.neighbors_memoized) == self.size()

    def new_value_satisfies_condition(self, x, y, value):
        condition = self.condition_array[x, y]
        if callable(condition):
            res = condition(value)
            assert type(res) in [bool, np.bool_], "invalid condition return value at {} for value {}: {} of type {}".format((x, y), value, res, type(res))
            return res
        else:
            raise ValueError("invalid condition type {}".format(type(condition)))

    def fill_position(self, x, y, value):
        assert (x, y) not in self.frozen_points, "can't change frozen point {}".format((x, y))
        self.array[x, y] = value

    def fill_point_set(self, point_set, value):
        for p in point_set:
            if p not in self.frozen_points:
                self.fill_position(p[0], p[1], value)

    def fill_all(self, value):
        for x in range(self.x_size):
            for y in range(self.y_size):
                self.fill_position(x, y, value)

    def get_value_at_position(self, x, y):
        return self.array[x, y]

    def get_neighbors(self, x, y, mode=8):
        if mode == 4:
            res = set()
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                p = (x+dx, y+dy)
                if self.is_valid_point(*p):
                # if 0 <= x+dx < self.x_size and 0 <= y+dy < self.y_size:
                    res.add(p)
            return res

        elif mode == 8:
            rep = self.get_representative_pixel(x, y)
            offset = (x - rep[0], y - rep[1])
            if rep in self.neighbors_memoized:
                rep_neighbors = self.neighbors_memoized[rep]
                return {(n[0] + offset[0], n[1] + offset[1]) for n in rep_neighbors}  # flagged as slow: setcomp
    
            res = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    if new_x < 0 or new_x >= self.x_size:
                        continue
                    if new_y < 0 or new_y >= self.y_size:
                        continue
                    res.add((new_x, new_y))
            self.neighbors_memoized[(x, y)] = res
            return res
        else:
            raise ValueError("unknown mode {}".format(mode))

    def get_random_path(self, a, b, points_to_avoid):
        # start and end should inch toward each other
        i = 0
        points_in_path = {a, b}
        current_a = a
        current_b = b
        while True:
            which_one = i % 2
            current_point = [current_a, current_b][which_one]
            objective = [current_b, current_a][which_one]
            points_to_avoid_this_step = points_to_avoid | points_in_path
            
            next_step = self.get_next_step_in_path(current_point, objective, points_to_avoid_this_step)
            if which_one == 0:
                current_a = next_step
            elif which_one == 1:
                current_b = next_step
            else:
                raise

            points_in_path.add(next_step)

            if current_a in self.get_neighbors(*current_b):
                break

            i += 1
        return points_in_path

    def get_next_step_in_path(self, current_point, objective, points_to_avoid):
        dx = objective[0] - current_point[0]
        dy = objective[1] - current_point[1]
        z = dx + dy*1j
        objective_angle = np.angle(z, deg=True)
        neighbors = self.get_neighbors(*current_point)
        neighbors = [n for n in neighbors if n not in points_to_avoid]
        neighbor_angles = [np.angle((n[0]-current_point[0]) + (n[1]-current_point[1])*1j, deg=True) for n in neighbors]
        # set objective angle to zero for purposes of determining weight
        neighbor_effective_angles = [abs(a - objective_angle) % 360 for a in neighbor_angles]
        neighbor_effective_angles = [a-360 if a>180 else a for a in neighbor_effective_angles]
        neighbor_weights = [180-abs(a) for a in neighbor_effective_angles]
        assert all(0 <= w <= 180 for w in neighbor_weights), "{}\n{}\n{}".format(neighbors, neighbor_effective_angles, neighbor_weights)
        total_weight = sum(neighbor_weights)
        norm_weights = [w / total_weight for w in neighbor_weights]
        chosen_one_index = np.random.choice([x for x in range(len(neighbors))], p=norm_weights)
        chosen_one = neighbors[chosen_one_index]
        return chosen_one

    def get_random_contiguous_region(self, x=None, y=None, expected_size=None, points_to_avoid=None, prioritize_internal_unfilled=False):
        assert expected_size > 1, "invalid expected size {}".format(expected_size)
        if points_to_avoid is None:
            points_to_avoid = set()
        points_to_avoid |= self.frozen_points
        center = (x, y)
        while center is (None, None) or center in points_to_avoid:
            center = self.get_random_point()
        neighbors = [p for p in self.get_neighbors(*center) if p not in points_to_avoid]
        size = -1
        while size < 1:
            size = int(np.random.normal(expected_size, expected_size/2))
        # print("making region of size {}".format(size))

        radius = int(round(np.sqrt(size/np.pi)))
        circle = self.get_circle_around_point(center[0], center[1], radius, barrier_points=points_to_avoid)
        return circle

    def get_circle_around_point(self, x, y, radius, barrier_points=None):  # flagged as slow, look for existing algorithms
        # print("\ncenter {}\nbarrier points\n{}".format((x, y), sorted(barrier_points)))
        # input("please debug")
        if barrier_points is None:
            barrier_points = set()
        assert (x, y) not in barrier_points, "can't make circle with center in barrier"
        points = {(x, y)}
        def get_max_dy(dx):
            return int(round(np.sqrt(radius**2-dx**2)))

        # can tell which points are on inside vs outside of barrier wall by doing this:
        # starting in center, go left/right and count barrier crossings (+= 0.5 when is_in_barrier_set changes truth value)
        # then go up/down from there and continue counting
        # so each point in the set is associated with a number of barrier crossings
        # those ending in 0.5 are in the barrier itself, those == 1 mod 2 are on the other side
        # so take those == 0 mod 2

        barrier_crossings_by_point = {(x, y): 0}
        
        # print("\nstarting circle around {}".format((x, y)))
        # start with dx = 0 to create the central line
        dy = 0
        max_dx = radius
        for direction in [-1, 1]:
            # reset barrier crossings to center's value
            n_barrier_crossings = barrier_crossings_by_point[(x, y)]
            last_was_on_barrier = False
            for abs_dx in range(max_dx+1):
                dx = direction * abs_dx
                this_x = x + dx
                this_y = y + dy
                # print("adding central line point at {}".format((this_x, this_y)))
                if not self.is_valid_point(this_x, this_y):
                    continue
                is_on_barrier = (this_x, this_y) in barrier_points
                if is_on_barrier != last_was_on_barrier:
                    n_barrier_crossings += 0.5
                barrier_crossings_by_point[(this_x, this_y)] = n_barrier_crossings
                last_was_on_barrier = is_on_barrier

        # now do the rest
        for dx in range(-radius, radius+1):
            this_x = x + dx
            central_axis_point = (x, y)
            assert central_axis_point in barrier_crossings_by_point, "can't find central axis point {}".format(central_axis_point)
            max_dy = get_max_dy(dx)
            for direction in [-1, 1]:
                n_barrier_crossings = barrier_crossings_by_point[central_axis_point]
                last_was_on_barrier = central_axis_point in barrier_points
                for abs_dy in range(max_dy+1):
                    dy = direction * abs_dy
                    this_y = y + dy
                    # print("adding non-central point at {}".format((this_x, this_y)))
                    if not self.is_valid_point(this_x, this_y):
                        continue
                    is_on_barrier = (this_x, this_y) in barrier_points
                    if is_on_barrier != last_was_on_barrier:
                        n_barrier_crossings += 0.5
                    barrier_crossings_by_point[(this_x, this_y)] = n_barrier_crossings
                    last_was_on_barrier = is_on_barrier

        res = {p for p, n in barrier_crossings_by_point.items() if n % 2 == 0}
        # print("returning:\n{}".format(res))

        # for dx in range(-radius, radius+1):
        #     max_dy = get_max_dy(dx)
        #     for ?
        # starting_set = {(x+dx, y+dy) for dx in range(-radius, radius+1) for dy in get_dy_range(dx)}
        # print("before barrier len = {}".format(len(res)))
        # res = self.apply_barrier(res, barrier_points, x, y)
        # print(" after barrier len = {}".format(len(res)))
        return res

    def get_distances_from_edge(self, point_set, use_scipy_method=True):
        if use_scipy_method:
            min_x = np.inf
            max_x = -np.inf
            min_y = np.inf
            max_y = -np.inf
            for p in point_set:
                x, y = p
                min_x = min(x, min_x)
                max_x = max(x, max_x)
                min_y = min(y, min_y)
                max_y = max(y, max_y)
            
            # put them in an array
            to_relative_coords = lambda p: (p[0]-min_x, p[1]-min_y)
            to_absolute_coords = lambda p: (p[0]+min_x, p[1]+min_y)
            arr_x_size = max_x - min_x + 1
            arr_y_size = max_y - min_y + 1
            arr = np.zeros((arr_x_size, arr_y_size))
            rels = {}
            for p in point_set:
                xrel, yrel = to_relative_coords(p)
                rels[p] = (xrel, yrel)
                arr[xrel, yrel] = 1
            distance_transform_matrix = ndimage.morphology.distance_transform_edt(arr)
            res = {}
            for p in point_set:
                xrel, yrel = rels[p]
                d = distance_transform_matrix[xrel, yrel]
                res[p] = d - 1
            return res

        else:
            # old way, slower than scipy
            if len(point_set) == 0:
                return {}
            res = {}
            points_on_edge = [p for p in point_set if any(n not in point_set for n in self.get_neighbors(*p))]  # flagged as slow: genexpr
            assert len(points_on_edge) > 0, "point set has no edge members:\n{}".format(sorted(point_set))
            for p in points_on_edge:
                res[p] = 0
            interior_point_set = point_set - set(points_on_edge)
            if len(interior_point_set) > 0:
                interior_distances = self.get_distances_from_edge(interior_point_set, use_scipy_method=False)
                for p, d in interior_distances.items():
                    res[p] = d + 1
            return res

    def make_random_elevation_change(self, expected_size, positive_feedback=False):
        center = self.get_random_point()
        # center = self.get_random_untouched_point()
        # print("making change at {}".format(center))
        changing_reg = self.get_random_contiguous_region(center[0], center[1], expected_size=expected_size, points_to_avoid=self.frozen_points)
        changing_reg_size = len(changing_reg)
        # print("expected size {}, got {}".format(expected_size, changing_reg_size))
        changing_reg_center_of_mass = (
            int(round(1/changing_reg_size * sum(p[0] for p in changing_reg))),
            int(round(1/changing_reg_size * sum(p[1] for p in changing_reg)))
        )
        e_center_of_mass = self.array[changing_reg_center_of_mass[0], changing_reg_center_of_mass[1]]
        reference_x, reference_y = changing_reg_center_of_mass
        # radius_giving_equivalent_area = np.sqrt(changing_reg_size/np.pi)
        radius_giving_expected_area = np.sqrt(expected_size/np.pi)
        desired_area_ratio_at_sea_level = 5
        desired_area_ratio_at_big_abs = 0.1
        big_abs = 1000

        # try to get mountain chains to propagate:
        # if center point is low abs, look at bigger region, might catch mountain
        # if center point is high abs, look at smaller region, don't let lowland water it down
        if abs(e_center_of_mass) >= big_abs:
            desired_area_ratio = desired_area_ratio_at_big_abs
        else:
            slope = (desired_area_ratio_at_big_abs - desired_area_ratio_at_sea_level) / (big_abs - 0)
            desired_area_ratio = desired_area_ratio_at_sea_level + slope * abs(e_center_of_mass)
        # radius = int(round(radius_giving_equivalent_area * np.sqrt(desired_area_ratio)))
        radius = int(round(radius_giving_expected_area * np.sqrt(desired_area_ratio)))
        reference_reg = self.get_circle_around_point(reference_x, reference_y, radius=radius)
        # reference_reg = changing_reg
        distances = self.get_distances_from_edge(changing_reg)
        max_d = max(distances.values())
        if max_d == 0:
            raw_func = elevation_change_constant
        else:
            raw_func = random.choice(ELEVATION_CHANGE_FUNCTIONS)

        if positive_feedback:
            # land begets land, sea begets sea
            elevations_in_refreg = [self.array[p[0], p[1]] for p in reference_reg]
            e_avg = np.mean(elevations_in_refreg)
            e_max = np.max(elevations_in_refreg)  # for detecting mountain nearby, chain should propagate
            e_min = np.min(elevations_in_refreg)
            elevation_sign = (1 if e_avg > 0 else -1)
            big_signed = elevation_sign * big_abs
            critical_abs = 100  # above this abs, go farther in that direction until reach big_abs_elevation
            critical_signed = elevation_sign * critical_abs
            critical_excess = e_avg - critical_signed
            big_remainder = big_signed - e_avg
            mountain_or_trench_nearby = abs(e_max) >= big_abs or abs(e_min) >= big_abs
  
            mu = \
                0 if abs(e_avg) > big_abs else \
                10 if e_avg > critical_abs else \
                -10 if e_avg < -1*critical_abs else \
                0

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

        sigma = max(10, abs(e_avg)) if abs(e_avg) < big_abs else 10
        max_change = np.random.normal(mu, sigma)

        func = lambda d: raw_func(d, max_d, max_change)
        changes = {p: func(d) for p, d in distances.items() if p not in self.frozen_points}
        for p, d_el in changes.items():
            current_el = self.get_value_at_position(*p)
            new_el = current_el + d_el
            if self.new_value_satisfies_condition(p[0], p[1], new_el):
                self.fill_position(p[0], p[1], new_el)

    def get_random_point(self, border_width=0):
        x = random.randrange(border_width, self.x_size - border_width)
        y = random.randrange(border_width, self.y_size - border_width)
        return (x, y)

    def get_random_zero_loop(self):
        x0_0, y0_0 = self.get_random_point(border_width=2)
        dx = 0
        dy = 0
        while np.sqrt(dx**2 + dy**2) < self.x_size * 1/2 * np.sqrt(2):
            x0_1 = random.randrange(2, self.x_size - 2)
            y0_1 = random.randrange(2, self.y_size - 2)
            dx = x0_1 - x0_0
            dy = y0_1 - y0_0
        # x0_1 = (x0_0 + self.x_size // 2) % self.x_size  # force it to be considerably far away to get more interesting result
        # y0_1 = (y0_0 + self.y_size // 2) % self.y_size
        source_0 = (x0_0, y0_0)
        source_1 = (x0_1, y0_1)
        path_0 = self.get_random_path(source_0, source_1, points_to_avoid=set())
        path_1 = self.get_random_path(source_0, source_1, points_to_avoid=path_0)

        res = path_0 | path_1
        # print(res)
        return res

    def add_random_zero_loop(self):
        points = self.get_random_zero_loop()
        for p in points:
            self.fill_position(p[0], p[1], 0)

    def fill_elevations_outward_propagation(self):
        # old
        raise Exception("do not use")
        # try filling the neighbors in "shells", get all the neighbors at one step and do all of them first before moving on
        # but that leads to a square propagating outward, so should sometimes randomly restart the list
        while len(self.unfilled_neighbors) + len(self.unfilled_inaccessible) > 0:
            to_fill_this_round = [x for x in self.unfilled_neighbors]
            random.shuffle(to_fill_this_round)
            for p in to_fill_this_round:
                # p = random.choice(list(self.unfilled_neighbors))
                # neighbor = random.choice([x for x in self.get_neighbors(*p) if x in self.filled_positions])
                # neighbor_el = self.get_value_at_position(*neighbor)
                filled_neighbors = [x for x in self.get_neighbors(*p) if x in self.filled_positions]
                average_neighbor_el = np.mean([self.get_value_at_position(*n) for n in filled_neighbors])
                d_el = np.random.normal(0, 100)
                el = average_neighbor_el + d_el
                self.fill_position(p[0], p[1], el)
                if random.random() < 0.02:
                    break  # for
                    # then will have new list of neighbors
            self.draw()

    def fill_elevations(self, n_steps, expected_change_size, plot_every_n_steps=None):
        if plot_every_n_steps is not None:
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
            if i % 100 == 0:
                try:
                    dt = datetime.now() - t0
                    n_left = n_steps - i
                    secs_per_step = dt/i
                    eta = secs_per_step * n_left
                    eta_str = str(eta)
                    print("step {}, {} elapsed, {} ETA".format(i, dt, eta_str))
                except ZeroDivisionError:
                    print("div/0!")
            self.make_random_elevation_change(expected_change_size, positive_feedback=True)
            # print("now have {} untouched points".format(len(self.untouched_points)))
            if plot_every_n_steps is not None and i % plot_every_n_steps == 0:
                try:
                    self.draw()
                except ValueError:
                    print("skipping ValueError")
                # input("debug")
            i += 1

    def get_xy_meshgrid(self):
        return np.meshgrid(range(self.x_size), range(self.y_size))

    def get_latlon_meshgrid(self):
        lats_grid = np.array([[None for y in range(self.y_size)] for x in range(self.x_size)])
        lons_grid = np.array([[None for y in range(self.y_size)] for x in range(self.x_size)])
        n_ps = self.x_size * self.y_size
        i = 0
        for x in range(self.x_size):
            for y in range(self.y_size):
                if i % 1000 == 0:
                    print("i = {}/{}".format(i, n_ps))
                p_lat, p_lon = mcm.get_lat_lon_of_point_on_map(x, y, self.x_size, self.y_size,
                    self.lat00, self.lon00,
                    self.lat01, self.lon01,
                    self.lat10, self.lon10,
                    self.lat11, self.lon11,
                    deg=True
                )
                lats_grid[x, y] = p_lat
                lons_grid[x, y] = p_lon
                i += 1

        # x_grid, y_grid = self.get_xy_meshgrid()
        # f = lambda x, y: mcm.get_lat_lon_of_point_on_map(x, y, self.x_size, self.y_size,
        #             self.lat00, self.lon00,
        #             self.lat01, self.lon01,
        #             self.lat10, self.lon10,
        #             self.lat11, self.lon11,
        #             deg=True
        #         )
        # f = np.vectorize(f)
        # lats_grid, lons_grid = f(x_grid, y_grid)
        return lats_grid, lons_grid

    def plot(self, projection=None):
        plt.gcf()
        self.pre_plot(projection=projection)
        plt.show()

    def draw(self):
        plt.gcf().clear()
        self.pre_plot()
        plt.draw()
        plt.pause(0.001)

    def save_plot_image(self, output_fp):
        self.pre_plot()
        plt.savefig(output_fp)

    def pre_plot(self, projection=None):
        if projection is None:
            projection = "cyl"  # Basemap's default is "cyl", which is equirectangular
        average_lat, average_lon = self.average_latlon()
        m = Basemap(projection=projection, lon_0=average_lon, lat_0=average_lat, resolution='l')
        # m.drawcoastlines()  # Earth
        # m.fillcontinents(color='coral',lake_color='aqua')  # Earth
        # draw parallels and meridians.
        m.drawparallels(np.arange(-90.,91.,30.))
        m.drawmeridians(np.arange(-180.,181.,60.))
        # m.drawmapboundary(fill_color='aqua')
        # plt.title("Full Disk Orthographic Projection")
        # plt.show()
        # ax = plt.subplot(projection=projection)

        min_elevation = self.array.min()
        max_elevation = self.array.max()
        n_sea_contours = 20
        n_land_contours = 100
        if min_elevation < 0:
            sea_contour_levels = np.linspace(min_elevation, 0, n_sea_contours)
        else:
            sea_contour_levels = [0]
        if max_elevation > 0:
            land_contour_levels = np.linspace(0, max_elevation, n_land_contours)
        else:
            land_contour_levels = [0]
        assert sea_contour_levels[-1] == land_contour_levels[0] == 0
        contour_levels = list(sea_contour_levels[:-1]) + list(land_contour_levels)
        colormap = get_land_and_sea_colormap()
        # care more about seeing detail in land contour; displaying deep sea doesn't matter much
        max_color_value = max_elevation
        min_color_value = -1 * max_elevation
        X, Y = self.get_latlon_meshgrid()
        print("X =", X)
        print("Y =", Y)
        X, Y = m(X, Y)  # https://stackoverflow.com/questions/48191593/

        # draw colored filled contours
        m.contourf(X, Y, self.array, cmap=colormap, levels=contour_levels, vmin=min_color_value, vmax=max_color_value)
        try:
            plt.colorbar()
        except IndexError:
            # np being stupid when there are too few contours
            pass

        # draw contour lines, maybe just one at sea level
        m.contour(X, Y, self.array, levels=[min_elevation, 0, max_elevation], colors="k")

        plt.gca().invert_yaxis()
        plt.axis("scaled")  # maintain aspect ratio
        plt.title("elevation")
        # max_grad, pair = self.get_max_gradient()
        # p, q = pair
        # print("max gradient is {} from {} to {}".format(max_grad, p, q))

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
    def from_image(image_fp, color_condition_dict, default_color):
        # all points in image matching something in the color dict should be that color no matter what
        # everything else is randomly generated
        # i.e., put the determined points in points_to_avoid for functions that take it

        if any(len(x) != 4 for x in color_condition_dict.keys()):
            raise ValueError("all color keys must have length 4, RGBA:\n{}".format(color_condition_dict.keys()))

        lattice = LatitudeLongitudeLattice  # image is x-y grid

        im = Image.open(image_fp)
        width, height = im.size
        m = ElevationGenerationMap(height, width)  # rows, columns
        arr = np.array(im)
        color_and_first_seen = {}
        for x in range(m.x_size):
            for y in range(m.y_size):
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
                    m.fill_position(x, y, fill_value)
                    m.add_condition_at_position(x, y, condition)
                    if is_frozen:
                        m.freeze_point(x, y)
        return m

    @staticmethod
    def load_elevation_data(data_fp, latlon00, latlon01, latlon10, latlon11):
        with open(data_fp) as f:
            lines = f.readlines()
        lines = [[float(x) for x in line.split(",")] for line in lines]
        array = np.array(lines)
        x_size, y_size = array.shape
        return ElevationGenerationMap(x_size, y_size, latlon00, latlon01, latlon10, latlon11, array=array)

    def freeze_coastlines(self):
        coastal_points = set()
        for x in range(self.x_size):
            for y in range(self.y_size):
                if self.array[x, y] < 0:
                    neighbors = self.get_neighbors(x, y)
                    for n in neighbors:
                        if self.array[n[0], n[1]] >= 0:
                            coastal_points.add(n)
        for p in coastal_points:
            self.fill_position(p[0], p[1], 0)
            self.freeze_point(*p)

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

    def save_elevation_data(self, output_fp):
        # format is just grid of comma-separated numbers
        # if not confirm_overwrite_file:
        #     return
        open(output_fp, "w").close()  # clear file
        for x in range(self.x_size):
            this_row = ""
            for y in range(self.y_size):
                el = self.array[x, y]
                this_row += "{:.1f},".format(el)
            assert this_row[-1] == ","
            this_row = this_row[:-1] + "\n"
            open(output_fp, "a").write(this_row)
        print("finished saving elevation data to {}".format(output_fp))