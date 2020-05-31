import numpy as np
import matplotlib.pyplot as plt


def deg_to_rad(x):
    return x * np.pi / 180

def rad_to_deg(x):
    return x * 180 / np.pi

def angle_between_vectors(v1, v2):
    dot = np.dot(v1, v2)  # = mag(v1) * mag(v2) * cos(theta)
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    cos_theta = dot / (len1 * len2)
    return np.arccos(cos_theta)

def unit_vector_lat_lon_to_cartesian(lat, lon, deg=True):
    if deg:
        # got deg from user
        lat = deg_to_rad(lat)
        lon = deg_to_rad(lon)
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    assert abs(1-np.linalg.norm([x, y, z])) < 1e-6, "need unit vector"
    return np.array([x, y, z])

def unit_vector_cartesian_to_lat_lon(x, y, z, deg=True):
    # latlon [0, 0] maps to xyz [1, 0, 0] (positive x comes out of Gulf of Guinea)
    # latlon [0, 90deg] maps to xyz [0, 1, 0] (positive y comes out of Indian Ocean)
    assert abs(1-np.linalg.norm([x, y, z])) < 1e-6, "need unit vector"
    lat = np.arcsin(z)
    assert abs(np.cos(lat) - np.sqrt(1 - z**2)) < 1e-6, "math error in sin cos lat"
    # arcsin :: [-1, 1] -> [-pi/2 = -90deg , pi/2 = 90deg]  # always gives you front hemisphere
    # arccos :: [-1, 1] -> [pi = 180deg , 0 = 0deg]  # always gives you eastern hemisphere
    # cos(lat) is 0 at both poles and 1 at equator, always non-negative, so don't need to worry about abs of it
    # neither inverse function alone will unambiguously give you lon, since range of longitude is total of 360 deg
    # have to use knowledge of which quadrant you are in
    coslon = x / np.cos(lat)
    sinlon = y / np.cos(lat)
    # if x > 0, you're in the front hemisphere (centered on Africa), if x < 0, you're in the back hemisphere (centered on Pacific)
    # if y > 0, you're in the eastern hemisphere, if y < 0, you're in the western hemisphere
    # lon is angle from standard position pointing along positive x, visualize normal unit circle
    # sin(lon) is 0 at prime meridian and antimeridian, 1 in India, and -1 in South America
    # cos(lon) is 1 at prime meridian, -1 at antimeridian, and 0 in India and South America
    front_or_back = "front" if x >= 0 else "back"
    east_or_west = "east" if y >= 0 else "west"
    raw_arccos_lon = np.arccos(coslon)
    raw_arcsin_lon = np.arcsin(sinlon)
    if front_or_back == "back":
        # the arcsin will be wrong, need to reflect over +/- 90 deg
        if raw_arcsin_lon >= 0:
            arcsin_lon = 180 - raw_arcsin_lon
        else:
            arcsin_lon = -180 - raw_arcsin_lon
    else:
        arcsin_lon = raw_arcsin_lon
    if east_or_west == "west":
        # the arccos will be wrong, need to change eastern to western
        arccos_lon = -1 * raw_arccos_lon
    else:
        arccos_lon = raw_arccos_lon

    assert abs(arcsin_lon - arccos_lon) < 1e-6, "methods for correcting sin and cos longitudes do not agree:\nxyz = {} {} {}\nhemispheres = {} {}\nraw_arcsin_lon = {}, raw_arccos_lon = {}\narcsin_lon = {}, arccos_lon = {}".format(x, y, z, front_or_back, east_or_west, raw_arcsin_lon, raw_arccos_lon, arcsin_lon, arccos_lon)
    lon = arcsin_lon

    if front_or_back == "front":
        assert -90 <= lon <= 90
    else:
        assert -180 <= lon <= -90 or 90 <= lon <= 180
    if east_or_west == "east":
        assert 0 <= lon <= 180
    else:
        assert -180 <= lon <= 0
    
    # old way, mistakenly always maps negative x to positive
    # xy_dilation = np.cos(lat)
    # lon2 = np.arccos(x / xy_dilation)  # don't use arccos because its range is only positive
    # lon = np.arcsin(y / xy_dilation)
    # assert abs(lon - lon2) < 1e-6, "math error\nx {} y {} z {}\nlat {} lon {} lon2 {}".format(x, y, z, lat, lon, lon2)

    if deg:
        # must give deg to user
        lat = rad_to_deg(lat)
        lon = rad_to_deg(lon)
    return np.array([lat, lon])

def rotate_partially_toward_other_unit_vector(p, q, alpha):
    # print("p {} q {} alpha {}".format(p, q, alpha))
    # get vector starting from p and rotating alpha of the way towards q
    assert p.shape == q.shape == (3,), "must apply function to 3D vectors"
    assert abs(1-np.linalg.norm(p)) < 1e-6 and abs(1-np.linalg.norm(q)) < 1e-6, "must apply function to unit vectors"
    assert 0 <= alpha <= 1, "alpha must be between 0 and 1"
    if alpha == 0:
        return p
    if alpha == 1:
        return q
    angle_p_q = angle_between_vectors(p, q)
    angle_to_move = alpha * angle_p_q
    if angle_p_q == 0:
        assert p == q, "got zero angle for unequal vectors"
        return p
    if angle_p_q == np.pi:
        assert p == -1*q, "got pi angle for non-opposite vectors"
        raise ValueError("great circle direction is undefined for opposite vectors")
    # https://stackoverflow.com/questions/22099490/calculate-vector-after-rotating-it-towards-another-by-angle-%CE%B8-in-3d-space
    cross = np.cross(np.cross(p, q), p)
    cross_mag = np.linalg.norm(cross)
    D_tick = cross / cross_mag
    z = np.cos(angle_to_move)*p + np.sin(angle_to_move)*D_tick
    return z

def get_lat_lon_of_point_on_map(r, c, map_r_size, map_c_size,
                                map_r_min_c_min_lat, map_r_min_c_min_lon,
                                map_r_min_c_max_lat, map_r_min_c_max_lon,
                                map_r_max_c_min_lat, map_r_max_c_min_lon,
                                map_r_max_c_max_lat, map_r_max_c_max_lon,
                                deg=True):
    # like 3d printer
    # go down the rows alpha_r of the way first on left and right edge
    # then go alpha_c of the way across between those points
    alpha_r = r / map_r_size
    alpha_c = c / map_c_size
    p00 = unit_vector_lat_lon_to_cartesian(map_r_min_c_min_lat, map_r_min_c_min_lon, deg=deg)
    p01 = unit_vector_lat_lon_to_cartesian(map_r_min_c_max_lat, map_r_min_c_max_lon, deg=deg)
    p10 = unit_vector_lat_lon_to_cartesian(map_r_max_c_min_lat, map_r_max_c_min_lon, deg=deg)
    p11 = unit_vector_lat_lon_to_cartesian(map_r_max_c_max_lat, map_r_max_c_max_lon, deg=deg)

    pr0 = rotate_partially_toward_other_unit_vector(p00, p10, alpha_r)
    pr1 = rotate_partially_toward_other_unit_vector(p01, p11, alpha_r)
    prc = rotate_partially_toward_other_unit_vector(pr0, pr1, alpha_c)
    prc_lat_lon = unit_vector_cartesian_to_lat_lon(prc[0], prc[1], prc[2], deg=deg)
    return prc_lat_lon

if __name__ == "__main__":
    r_size = 2000
    c_size = 900
    r = 1100
    c = 880
    # lat00, lon00 = 51.5074, -0.1278  # London
    # lat01, lon01 = 60.1699, 24.9384  # Helsinki
    # lat10, lon10 = 40.4168, -3.7038  # Madrid
    # lat11, lon11 = 41.0082, 28.9784  # Istanbul
    lat00, lon00 = -33.9249, 18.4241  # Cape Town
    lat01, lon01 = -31.9505, 115.8605  # Perth
    lat10, lon10 = -54.8019, -68.3030  # Ushuaia
    lat11, lon11 = -41.2865, 174.7762  # Wellington
    p = get_lat_lon_of_point_on_map(r, c, r_size, c_size,
        lat00, lon00, lat01, lon01, lat10, lon10, lat11, lon11, deg=True)
    # print(p)

    plt.subplot(111, projection="mollweide")
    p = plt.plot([-1, 1, 1], [-1, -1, 1], "o-")
    plt.grid(True)

    plt.show()
