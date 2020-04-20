# GeoPandas example
# https://towardsdatascience.com/lets-make-a-map-using-geopandas-pandas-and-matplotlib-to-make-a-chloropleth-map-dddc31c1983d
# http://geopandas.org/gallery/plotting_with_geoplot.html, but cartopy is fucked up

import geopandas as gpd
# import geoplot
import matplotlib
import matplotlib.pyplot as plt
from shapely import geometry
import sys
import random
import numpy as np
import networkx as nx
from descartes import PolygonPatch


def plot_random_countries(world):
    row_selector = [random.random() < 0.5 for r_i in range(world.shape[0])]
    rows = world.ix[row_selector,]
    rows.plot()
    plt.show()

def plot_test_polygon():
    coords_list = [
        (0, 0), (0, 1), (3, 1.7), (1, 4.2), (-2, 2), (-1.4, -1.8),
    ]
    point_list = [geometry.Point(x) for x in coords_list]
    polygon = geometry.Polygon([[p.x, p.y] for p in point_list])
    names = ["Country1"]
    borders = [polygon]
    df = gpd.GeoDataFrame(columns=["country", "geometry"])
    for r_i in range(len(names)):
        df.ix[r_i, "country"] = names[r_i]
        df.ix[r_i, "geometry"] = borders[r_i]
    df.plot()
    plt.show()

def plot_outlines(world):
    world.plot(figsize=(8,8))#, edgecolor=u'gray')

def plot_border_distances(world, distance_matrix, reference_country_name):
    plot_outlines(world)
    # print(distance_matrix["Egypt"]["Libya"])
    distances = distance_matrix[reference_country_name]
    # print(distances)
    max_distance = max(distances.values())
    min_distance = 0
    # cmap = matplotlib.cm.get_cmap('Spectral')
    colors = {0: (1, 0, 0, 1)}
    r_increment = random.uniform(1/4, 1/3) * random.choice([-1, 1])
    g_increment = random.uniform(1/4, 1/3) * random.choice([-1, 1])
    b_increment = random.uniform(1/4, 1/3) * random.choice([-1, 1])
    for i in range(1, max_distance + 1):
        # colors[i] = (random.random(), random.random(), random.random(), 1)
        r, g, b, a = colors[i-1]
        r = (r + r_increment) % 1
        g = (g + g_increment) % 1
        b = (b + b_increment) % 1
        colors[i] = (r, g, b, a)
    for index, row in world.iterrows():
        country_name = row["name"]
        if country_name in distances:
            distance = distances[country_name]
            # distance_01 = distance / max_distance
            # color = cmap(distance_01)
            color = colors[distance]
        else:
            color = "black"
        # row.plot(color=color)

        # https://stackoverflow.com/questions/53142563/coloring-specific-countries-with-geopandas
        df_row = world[world.name == row["name"]]
        row_gm = df_row.__geo_interface__['features']  # geopandas's geo_interface
        row_g0 = {
            'type': row_gm[0]['geometry']['type'],
            'coordinates': row_gm[0]['geometry']['coordinates']
        }
        plt.gca().add_patch(PolygonPatch(row_g0,
            fc=color,
            # ec="black",  # edge color
            alpha=1,
            # zorder=2
        ))

    # plt.colorbar()
    plt.show()

def add_neighbors_column(world):
    # https://gis.stackexchange.com/questions/281652/find-all-neighbors-using-geopandas/281676
    df = world
    df["NEIGHBORS"] = None  # add NEIGHBORS column
    delim = ";"
    for index, country in df.iterrows():
        assert delim not in country["name"]
        # get 'not disjoint' countries
        neighbors = df[~df.geometry.disjoint(country.geometry)]["name"].tolist()
        neighbors = [name for name in neighbors if country["name"] != name]
        df.at[index, "NEIGHBORS"] = delim.join(neighbors)
    return df

def create_distance_matrix(world):
    if "NEIGHBORS" not in world.columns:
        world = add_neighbors_column(world)
    n_rows = world.shape[0]
    # print("there are {} countries in the world".format(n_rows))
    g = nx.Graph()
    for index, row in world.iterrows():
        g.add_node(row["name"])
    delim = ";"
    for index, row in world.iterrows():
        neighbors = row["NEIGHBORS"].split(delim)
        for n in neighbors:
            g.add_edge(row["name"], n)
    gen = nx.all_pairs_shortest_path_length(g)
    # print(type(gen))
    # for x in distance_matrix:
    #     print(type(x), x)
    #     input("a")
    return dict(gen)


if __name__ == "__main__":
    world_fp = gpd.datasets.get_path("naturalearth_lowres")
    world = gpd.read_file(world_fp)
    print(sorted(world.name))
    # print(world.columns)
    # plot_random_countries(world)
    distance_matrix = create_distance_matrix(world)
    reference_country_name = random.choice(world.name)
    plot_border_distances(world, distance_matrix, reference_country_name)
