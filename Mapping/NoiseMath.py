import random
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import MapCoordinateMath as mcm


def change_globe(df, key_str):
    n_steps = 5
    for i in range(n_steps):
        print("step {}/{}".format(i, n_steps))
        df = change_globe_circles(df, key_str)
        df = change_globe_waves(df, key_str)
        df = change_globe_spikes(df, key_str)
    return df


def change_globe_spikes(df, key_str):
    n_spikes = random.randint(100, 5000)
    sigma = 100
    df = add_random_data_spikes(df, key_str, n_spikes=n_spikes, sigma=sigma)
    return df


def change_globe_circles(df, key_str):
    n_patches = random.randint(50, 200)
    area_proportion_per_patch = 1/random.randint(50, 500)
    df = add_random_data_circles(df, key_str, n_patches=n_patches, area_proportion_per_patch=area_proportion_per_patch)
    return df


def change_globe_waves(df, key_str):
    n_waves = random.randint(10, 50)
    expected_amplitude = random.uniform(50, 150)
    # freq_sigma = random.uniform(10, 100) # freq of wave drawn from abs of norm(0, sigma), recall that radius of sphere is 1 so freq of 1 will have 1 period over the whole sphere
    df = add_random_data_radial_waves(df, key_str, n_waves=n_waves, expected_amplitude=expected_amplitude)
    return df


def get_random_wave_function_1d():
    # n_sins = 10  # just do 1 per call
    # shape = (n_sins,)
    amp    = get_random_sin_amp()
    offset = get_random_sin_offset()
    freq   = get_random_sin_freq()
    phase  = get_random_sin_phase()
    # def f(x, amp=amp, offset=offset, freq=freq, phase=phase):
    #     return (amp * np.sin(freq*x + phase) + offset).sum()
    return lambda x: amp * np.sin(freq*x + phase) + offset


def get_random_sin_amp(shape=None):
    return np.random.normal(0, 1, shape)


def get_random_sin_offset(shape=None):
    return np.random.normal(0, 1, shape)


def get_random_sin_freq(shape=None, max_freq=100):
    a = 0.5  # power law parameter, > 1 means higher numbers will occur more ofter
    res_in_01 = np.random.power(a, shape)
    return max_freq * res_in_01


def get_random_sin_phase(shape=None):
    return np.random.uniform(0, 2*np.pi, shape)


def get_sigmoid_decay_function(d_val_max, r_max, h_stretch_param=0):
    # f(0) is d_val_max (can be negative
    # f(1) is zero
    # r is scaled to [0,1], which can then be warped to change decay speed (using h parameter, higher h means function treats r as higher faster, so faster r growth at beginning, meaning sigmoid decays more rapidly; lower h does the opposite, postpones r growth until end, sigmoid decays more slowly)
    def f(r, r_max=r_max, h_stretch_param=h_stretch_param, d_val_max=d_val_max):
        r_01 = r/r_max
        r_01_stretched = transform_01_hyperbolic(r_01, h_stretch_param)
        # new_r = r_max * r_01_stretched
        return d_val_max/2 * (1 + np.cos(np.pi * r_01_stretched))
    # the resulting d_val is what you'll add to the data on the map
    return f


def test_plot_get_sigmoid_decay_function():
    d_val_max = 40
    r_max = 4
    h_stretch_params = [0.4, 0.25, 0, -0.25, -0.4]
    rs = np.linspace(0, r_max, 100)
    for h in h_stretch_params:
        func = get_sigmoid_decay_function(d_val_max, r_max, h)
        d_vals = func(rs)
        plt.plot(rs, d_vals)
    plt.show()


def transform_01_hyperbolic(x, h):
    # this maps [0, 1] to itself one-to-one, with slower growth at beginning and faster growth at end or vice versa
    # h is parameter of stretching, min -1, max 1
    # for h = 0, just returns y=x
    # for h > 0, the point (0.5, 0.5) is pulled toward (0, 1) (upper left)
    # for h < 0, the point (0.5, 0.5) is pulled toward (1, 0) (lower right)
    # the "midpoint" (along arc length) is thus ((1-h)/2 , (1+h)/2)
    # hyperbola through the midpoint and (0,0) and (1,1), from three-point formula (wikipedia and wolfram, verified on desmos) is:
    # y = (x*(h+1)^2) / (h^2 + h*(4*x-2) + 1)
    return (x * (h+1)**2) / (h**2 + h*(4*x-2) + 1)


def test_plot_transform_01_hyperbolic():
    hs = [-0.75, -0.5, 0, 0.5, 0.75]
    xs = np.linspace(0, 1, 100)
    for h in hs:
        ys = transform_01_hyperbolic(xs, h)
        plt.plot(xs, ys)
    plt.show()


def get_area_proportions_power_law(n_samples):
    a = 0.25  # power law shape parameter (< 1 means lower numbers are more common)
    return np.random.power(a, size=(n_samples,))


def add_random_data_independent_all_points(df, key_str, n_iterations, sigma):
    n_points = len(df.index)
    if key_str not in df.columns:
        df[key_str] = np.zeros((n_points,))
    for i in range(n_iterations):
        d_val_series = np.random.normal(0, sigma, n_points)
        df[key_str] += d_val_series
    return df


def add_random_data_spikes(df, key_str, n_spikes, sigma):
    if key_str not in df.columns:
        df[key_str] = np.zeros((len(df.index),))
    for i in range(n_spikes):
        p_i = random.choice(df.index)
        d_val = np.random.normal(0, sigma)
        df.loc[p_i, key_str] += d_val
    return df


def add_random_data_radial_waves(df, key_str, n_waves, expected_amplitude):
    print("adding {} radial waves of variable {}".format(n_waves, key_str))
    if key_str not in df.columns:
        df[key_str] = np.zeros((len(df.index),))
    for i in range(n_waves):
        if i % 100 == 0:
            print("i = {}/{}".format(i, n_waves))
        f = get_random_wave_function_1d()
        # just do radius in 3d for now, don't care to convert it to sphere path right now
        # determine amplitude roughly
        sample_rs = np.linspace(0, 1, 100)
        sample_amp = max(abs(f(sample_rs)))
        multiplier = expected_amplitude / sample_amp
        starting_p_i = random.choice(df.index)
        starting_xyz = df.loc[starting_p_i, "xyz"]
        starting_xyz_array = np.tile(starting_xyz, (len(df.index), 1))  # == np.array(starting_xyz for i in range(self.n_points()))
        xyzs = np.stack(df["xyz"].values)
        dxyzs = (xyzs - starting_xyz_array) ** 2
        distances = np.sqrt(dxyzs.sum(axis=1))
        vals = f(distances) * multiplier
        df[key_str] += vals
    return df


def add_random_data_circles(df, key_str, n_patches, area_proportions=None, mu_colname=None, sigma_colname=None, expectation_colname=None, expectation_omega_colname=None):
    print("adding {} circles of variable {}".format(n_patches, key_str))
    if area_proportions is None:
        area_proportions = get_area_proportions_power_law(n_patches)
    assert len(area_proportions) == n_patches
    if key_str not in df.columns:
        df[key_str] = np.zeros((len(df.index),))

    for i in range(n_patches):
        area_proportion = area_proportions[i]
        radius_3d = mcm.get_radius_about_center_surface_point_for_circle_of_area_proportion_on_unit_sphere(area_proportion)
        if i % 100 == 0:
            print("i = {}/{}".format(i, n_patches))
        starting_p_i = random.choice(df.index)
        starting_xyz = df.loc[starting_p_i, "xyz"]
        starting_xyz_array = np.tile(starting_xyz, (len(df.index), 1))  # == np.array(starting_xyz for i in range(self.n_points()))
        xyzs = np.stack(df["xyz"].values)
        dxyzs = (xyzs - starting_xyz_array) ** 2
        distances = np.sqrt(dxyzs.sum(axis=1))
        in_region_mask = pd.Series(distances <= radius_3d)
        in_region_mask_index = df.index[in_region_mask]  # translate from enumerated terms to df's index terms (e.g. have a bunch of random point numbers that aren't just 1..n)

        mu = 0 if mu_colname is None else df.loc[starting_p_i, mu_colname]
        sigma = 100 if sigma_colname is None else df.loc[starting_p_i, sigma_colname]
        expectation = 0 if expectation_colname is None else df.loc[starting_p_i, expectation_colname]  # what value should it tend toward at this point
        expectation_omega = 0 if expectation_omega_colname is None else df.loc[starting_p_i, expectation_omega_colname]  # how much of the discrepancy between the value and the expectation should go into the mu
        assert 0 <= expectation_omega <= 1
        discrepancy_from_expectation = df.loc[starting_p_i, key_str] - expectation
        mu += -1 * expectation_omega * discrepancy_from_expectation
        d_val = np.random.normal(mu, sigma)
        df.loc[in_region_mask_index, key_str] += d_val
    
    df = control_for_condition_ranges(df, key_str)
    return df


def control_for_condition_ranges(df, key_str):
    print(f"adjusting deviations in {key_str}")
    min_val_colname = f"min_{key_str}"
    max_val_colname = f"max_{key_str}"
    if min_val_colname not in df.columns:
        print(f"control_for_condition_ranges found no min for {key_str}")
        df[min_val_colname] = np.array([np.nan for i in range(len(df.index))])
    else:
        df[min_val_colname] = df[min_val_colname].fillna(np.nan)  # make sure it's a datatype we can work with, not None
    if max_val_colname not in df.columns:
        print(f"control_for_condition_ranges found no max for {key_str}")
        df[max_val_colname] = np.array([np.nan for i in range(len(df.index))])
    else:
        df[max_val_colname] = df[max_val_colname].fillna(np.nan)  # make sure it's a datatype we can work with, not None

    # do some kind of smooth surface addition (e.g. cubic spline over the whole map) to match the conditions
    deviation_from_min = df[key_str] - df[min_val_colname]
    deviation_from_max = df[key_str] - df[max_val_colname]
    has_min = ~pd.isna(df[min_val_colname])
    has_max = ~pd.isna(df[max_val_colname])
    has_min_and_max = has_min & has_max
    has_min_only = has_min & ~has_max
    has_max_only = has_max & ~has_min
    has_neither_min_nor_max = ~has_min & ~has_max
    assert has_min_only.sum() + has_max_only.sum() + has_min_and_max.sum() + has_neither_min_nor_max.sum() == len(df.index)
    # print(f"{has_min_only.sum()} min only, {has_max_only.sum()} max only, {has_min_and_max.sum()} both min and max, {has_neither_min_nor_max.sum()} neither min nor max")

    deviations = pd.Series(data=[np.nan for i in range(len(df.index))], index=df.index)
    # if has min only, the deviation is negative if below it and 0 otherwise
    deviations[has_min_only] = np.minimum(0, deviation_from_min)
    # if has max only, the deviation is positive if above it and 0 otherwise
    deviations[has_max_only] = np.maximum(0, deviation_from_max)
    # if has both max and min, deviation is 0 if between them, otherwise the deviation from the closer one
    deviations[has_min_and_max & (deviation_from_min <= 0)] = deviation_from_min
    deviations[has_min_and_max & (deviation_from_max >= 0)] = deviation_from_max
    deviations[has_min_and_max & (deviation_from_min >= 0) & (deviation_from_max <= 0)] = 0

    if pd.isna(deviations).all():
        # no change to be made
        pass
    else:
        assert np.isfinite(df[key_str]).all(), f"df[{key_str}] not all finite, before adjustment"
        non_na_deviations = deviations[~pd.isna(deviations)]
        # interpolate in xyz coords, not latlon, so it won't think points near poles are far apart
        # adjustment = get_interpolated_adjustment_for_condition_values(non_na_deviations, df)
        adjustment = get_trivial_adjustment_for_condition_values(non_na_deviations, df)  # debug
        assert np.isfinite(adjustment).all(), "adjustment not all finite"
        df[key_str] += adjustment
        assert np.isfinite(df[key_str]).all(), f"df[{key_str}] not all finite, after adjustment"

    assert meets_conditions(df, key_str), "failed to adjust df correctly"
    print(f"done adjusting deviations in {key_str}")
    return df


def meets_conditions(df, key_str):
    min_val_colname = f"min_{key_str}"
    max_val_colname = f"max_{key_str}"
    x = df[key_str]
    min_is_na = pd.isna(df[min_val_colname])
    meets_min = min_is_na | (x >= df[min_val_colname])
    max_is_na = pd.isna(df[max_val_colname])
    meets_max = max_is_na | (x <= df[max_val_colname])
    meets_both = meets_min & meets_max
    return meets_both.all()


def get_trivial_adjustment_for_condition_values(deviations, df):
    # just adjust any point with a deviation by the negative of that value, and leave all others alone
    adjustment = pd.Series(data=np.zeros((len(df.index),)), index=df.index)
    adjustment[deviations.index] = -1 * deviations
    return adjustment


def get_adjustment_for_condition_values(deviations, df):
    data_xyzs = df.loc[deviations.index, "xyz"]
    data_xyzs = np.array([np.array(tup) for tup in data_xyzs])
    assert data_xyzs.shape == (len(deviations.index), 3), data_xyzs.shape

    target_point_index = [x for x in df.index if x not in deviations.index]  # let the points with deviation values be adjusted by exactly those values, interpolate adjustment for everything else
    target_xyzs = np.array([np.array(tup) for tup in df.loc[target_point_index, "xyz"]])
    data_values = deviations.array
    # despite the name, scipy.griddata can interpolate from arbitrary unstructured data points to arbitrary unstructured target points
    # cubic spline doesn't work for more than 2D space
    interpolated = scipy.interpolate.griddata(points=data_xyzs, values=data_values, xi=target_xyzs, method="linear", fill_value=0)

    interpolated = pd.Series(data=interpolated, index=target_point_index)
    interpolated = pd.concat([interpolated, deviations])  # add back the actual deviation values for points that have them
    adjustment = -1 * interpolated
    return adjustment


def add_random_data_sigmoid_decay_hills(df, key_str, n_hills, h_stretch_parameters=None, mu_colname=None, sigma_colname=None):
    # mu and sigma are used to roll the d_val_max
    print("adding {} sigmoid decay hills of variable {}".format(n_hills, key_str))
    if h_stretch_parameters is None:
        h_stretch_parameters = np.random.uniform(-1, 1, n_hills)
    assert len(h_stretch_parameters) == n_hills
    if key_str not in df.columns:
        df[key_str] = np.zeros((len(df.index),))
    for i in range(n_hills):
        if i % 100 == 0:
            print("i = {}/{}".format(i, n_hills))
        # should make function for this, TODO
        starting_p_i = random.choice(df.index)
        starting_xyz = df.loc[starting_p_i, "xyz"]
        starting_xyz_array = np.tile(starting_xyz, (len(df.index), 1))
        xyzs = np.stack(df["xyz"].values)
        dxyzs = (xyzs - starting_xyz_array) ** 2
        distances = np.sqrt(dxyzs.sum(axis=1))

        h = h_stretch_parameters[i]
        mu = 0 if mu_colname is None else df.loc[starting_p_i, mu_colname]
        sigma = 100 if sigma_colname is None else df.loc[starting_p_i, sigma_colname]
        d_val_max = np.random.normal(mu, sigma)
        r_max = 2  # all the way across unit sphere
        sigmoid_func = get_sigmoid_decay_function(d_val_max, r_max, h)

        d_vals = sigmoid_func(distances)
        df[key_str] += d_vals
    return df


def add_random_data_jagged_patches(df, key_str, adjacencies, usp_to_index_function, n_patches, area_proportions=None):
    print("adding {} jagged patches of variable {}".format(n_patches, key_str))
    if area_proportions is None:
        area_proportions = get_area_proportions_power_law(n_patches)
    assert len(area_proportions) == n_patches
    if key_str not in df.columns:
        df[key_str] = np.zeros((len(df.index),))
    for i in range(n_patches):
        if i % 100 == 0:
            print("i = {}/{}".format(i, n_patches))
        starting_p_i = random.choice(range(len(df.index)))
        starting_point = df.loc[starting_p_i, "usp"]
        # print("starting point: {}".format(starting_point))
        patch_indices = {starting_p_i}
        patch_points = {starting_point}
        # the outward-moving edge is the next points that are not yet in the patch
        edge = set(adjacencies[starting_point])
        area_proportion = area_proportions[i]
        patch_size = int(area_proportion * len(df.index))
        for p_i in range(patch_size):
            chosen_point = random.choice(list(edge))
            # print("chosen: {}".format(chosen))
            chosen_p_i = usp_to_index_function(chosen_point)
            patch_points.add(chosen_point)
            patch_indices.add(chosen_p_i)
            edge |= set(adjacencies[chosen_point])
            edge -= patch_points
            if len(edge) == 0:  # can happen if whole lattice is in patch
                break
        # change values on the patch
        d_val = random.uniform(-100, 100)
        df.loc[patch_indices, key_str] += d_val
    return df


