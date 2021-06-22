# idea: pick one point at a time, at random
# look at its neighbors, get their average, use that as the expected value, add some variance
# for any neighbor which doesn't yet have a value, pretend it does (assign it a random value)


import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import networkx as nx


def get_noise_using_pseudo_neighbor_values(shape, neighbor_method):
    arr = np.zeros(shape)
    has_value = np.zeros(shape).astype(bool)
    coords = get_coords(shape, shuffle=True)

    if neighbor_method == "d8":
        neighbor_dict = get_d8_neighbor_dict_toroidal(shape)
    elif neighbor_method == "random_graph":
        neighbor_dict = get_random_graph_neighbor_dict(shape, expected_size=50)
    else:
        raise ValueError(f"unknown neighbor method: {neighbor_method}")


    for i,j in coords:
        neighbors = neighbor_dict[(i,j)]
        neighbor_values = []
        # get actual values to use as mean for the pseudo-values
        # then use the mean of all of these as the mean for the new value
        for ni,nj in neighbors:
            if has_value[ni,nj]:
                neighbor_values.append(arr[ni,nj])

        if len(neighbor_values) == 0:
            real_mean = 0
        else:
            real_mean = np.mean(neighbor_values)

        # make pseudo-values
        b = 1  # flatness of laplace distribution, b=1 is sharp, sharper with lower b
        for ni,nj in neighbors:
            if not has_value[ni,nj]:
                pseudo_value = np.random.laplace(real_mean, b)
                neighbor_values.append(pseudo_value)

        mean_to_use = np.mean(neighbor_values)  # including pseudo-values
        val = np.random.laplace(mean_to_use, b)

        # set it and mark that the point now has a real value
        arr[i,j] = val
        has_value[i,j] = True

    return arr


def get_coords(shape, shuffle):
    coords = [(i,j) for i in range(shape[0]) for j in range(shape[1])]
    if shuffle:
        random.shuffle(coords)
    return coords


def get_d8_neighbor_dict_toroidal(shape):
    d = {}
    for i,j in get_coords(shape, shuffle=False):
        d[(i,j)] = get_d8_neighbors_toroidal(i, j, shape)
    return d


def get_d8_neighbors_toroidal(i, j, shape):
    idown = (i-1) % shape[0]
    iup = (i+1) % shape[0]
    jdown = (j-1) % shape[1]
    jup = (j+1) % shape[1]
    neighbors = [
        (idown, jdown), (idown, j), (idown, jup),
        (i, jdown), (i, jup),
        (iup, jdown), (iup, j), (iup, jup),
    ]
    return neighbors


def get_random_graph_neighbor_dict(shape, expected_size):
    print("getting graph neighbor dict")
    # for each point, get a contiguous region around it by doing random walks, and make the point adjacent to all of those points (except itself)
    coords = get_coords(shape, shuffle=True)  # shuffle True is only necessary for when debug-plotting the regions just to see what they look like (so they're not all at the top of the graph); the algorithm doesn't care what order these coords are in
    neighbors_d8 = get_d8_neighbor_dict_toroidal(shape)
    # use d8 as the way to walk around through the space when finding the region to be neighbors with
    adjacency = {coord: set() for coord in coords}

    count = 0
    len_coords = len(coords)
    for i,j in coords:
        if count % 1000 == 0:
            print(f"count {count}/{len_coords}")
        region_coords = get_contiguous_region(i, j, neighbors_d8, expected_size, exclude_source=True)
        # plot_region(region_coords, shape)  # debug
        for y,z in region_coords:
            adjacency[(i,j)].add((y,z))
            adjacency[(y,z)].add((i,j))
        count += 1
    
    print("done getting graph neighbor dict")
    return adjacency


def plot_region(region_coords, shape):
    arr = np.zeros(shape)
    for i,j in region_coords:
        arr[i,j] = 1
    plt.imshow(arr)
    plt.show()


def get_contiguous_region(i, j, random_walk_neighbors, expected_size, exclude_source):
    res = [(i,j)]
    res += random_walk_neighbors[(i,j)]  # must have at least the default neighbors
    points_with_no_available_neighbors = {(i,j)}

    # some optimization so we don't have to search the region to know if a point is already in it or not
    res_set = set(res)

    while True:
        if random.random() < 1/expected_size:
            break
            # e.g. if expected_size is 5, 1/es is 0.2, so this condition will be met after expected 5 trials, and so growth will stop

        # find a point within the region from which growth can still occur
        # this is potentially very slow as it keeps choosing things in the interior of the region, will be more of a problem with larger expected_size
        while True:
            start_point = random.choice(res)
            if start_point in points_with_no_available_neighbors:
                continue
            candidates = random_walk_neighbors[start_point]
            available_candidates = [c for c in candidates if c not in res_set]
            if len(available_candidates) == 0:
                points_with_no_available_neighbors.add(start_point)
                continue
            else:
                break

        # now we have a non-empty list of available candidates, add one of these to the region
        chosen = random.choice(available_candidates)
        res.append(chosen)
        res_set.add(chosen)  # asking if point is in the region should take constant time

    if exclude_source:
        res.remove((i,j))

    return res


def get_nx_graph(adjacency_dict):
    g = nx.Graph()
    g.add_nodes_from(adjacency_dict.keys())
    for node, neighbors in adjacency_dict.items():
        edges_lst = [(node, neighbor) for neighbor in neighbors]
        g.add_edges_from(edges_lst)
    return g


def plot_distance_from_point(i, j, graph, shape, coords):
    arr = get_array_of_distances_from_point(i, j, graph, shape, coords)
    plt.imshow(arr)
    plt.colorbar()
    plt.title("distances to a point in a random-regionalized neighbor graph")
    plt.show()


def get_array_of_distances_from_point(i, j, graph, shape, coords):
    #  distance_matrix = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(graph, nodelist=coords)
    # node_index = coords.index((i,j))
    # distances = distance_matrix[node_index]

    distances = nx.single_source_shortest_path_length(graph, (i,j))  # no need to compute entire matrix for only one point

    # now create array where the Z value is the distance from (i,j)
    arr = np.zeros(shape)
    for coord_i, coord in enumerate(coords):
        y,z = coord
        arr[y,z] = distances[(y,z)]  # since distance matrix's nodelist is in same order as coords (passing nodelist=coords when making it)

    return arr


def get_noise_using_circles_in_graph(n_circles, graph, shape, coords):
    print("adding circle noise")
    # simple circle addition algorithm but using arbitrary adjacency
    arr = np.zeros(shape)
    for circle_i in range(n_circles):
        if circle_i % 10 == 0:
            print(f"circle {circle_i}/{n_circles}")
        i,j = random.choice(coords)
        distances = get_array_of_distances_from_point(i, j, graph, shape, coords)  # alternative here is to get the whole distance matrix; computation/memory tradeoff
        radius = np.exp(np.random.normal()) * 3
        circle = distances <= radius
        dz = np.random.normal()
        arr[circle] += dz
    print("done adding circle noise")
    return arr


if __name__ == "__main__":
    shape = (100, 100)
    coords = get_coords(shape, shuffle=False)

    random_region_adjacency_dict = get_random_graph_neighbor_dict(shape, expected_size=10)
    graph = get_nx_graph(random_region_adjacency_dict)
    # plot_distance_from_point(50,50, graph, shape, coords)

    # arr = get_noise_using_pseudo_neighbor_values(shape, neighbor_method="random_graph")
    arr = get_noise_using_circles_in_graph(100, graph, shape, coords)

    # plt.subplot(1,2,1)
    # plt.imshow(arr, origin="lower")
    # plt.colorbar()

    # plt.subplot(1,2,2)
    arr2 = convolve2d(arr, np.array([[1,1,1],[1,0,1],[1,1,1]]), mode="same", boundary="wrap")
    cf = plt.contourf(arr2)
    plt.contour(arr2, colors="k")
    plt.colorbar(cf)
    plt.axis("equal")
    plt.show()
