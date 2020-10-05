import random
import numpy as np
import pandas as pd
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


def get_area_proportions_power_law(n_samples):
    a = 0.25  # power law shape parameter (< 1 means lower numbers are more common)
    return np.random.power(a, size=(n_samples,))


def add_random_data_spikes(df, key_str, n_spikes, sigma):
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


def add_random_data_circles(df, key_str, n_patches, area_proportions=None, mu_colname=None, sigma_colname=None):
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
        mu = 0 if mu_colname is None else df.loc[starting_p_i, mu_colname]
        sigma = 100 if sigma_colname is None else df.loc[starting_p_i, sigma_colname]
        d_val = np.random.normal(mu, sigma)
        df.loc[in_region_mask, key_str] += d_val
    return df


def add_random_data_jagged_patches(df, key_str, adjacencies, usp_to_index, n_patches, area_proportions=None):
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
            chosen_p_i = usp_to_index[chosen_point]
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


