# GeoPandas example
# https://towardsdatascience.com/lets-make-a-map-using-geopandas-pandas-and-matplotlib-to-make-a-chloropleth-map-dddc31c1983d
# http://geopandas.org/gallery/plotting_with_geoplot.html, but cartopy is fucked up

import geopandas as gpd
# import geoplot
import matplotlib.pyplot as plt
from shapely import geometry
import sys
import random

# fp = "LondonMapExample/London_Borough_Excluding_MHW.shp"
# map_df = gpd.read_file(fp)# check data type so we can see that this is not a normal dataframe, but a GEOdataframe

world_fp = gpd.datasets.get_path("naturalearth_lowres")
print(world_fp)
world = gpd.read_file(world_fp)
# world.plot()
# plt.show()
# print(world.head())

row_selector = [random.random() < 0.5 for r_i in range(world.shape[0])]
rows = world.ix[row_selector,]
rows.plot()
plt.show()
# geom = row.get("geometry")
# print(geom)

sys.exit()

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
