import pandas as pd


def get_distance_matrix():
    borders = {}
    with open("StateAdjacency.txt") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        state, *neighbors = line.split(",")
        borders[state] = neighbors
    states = sorted(borders.keys())
    matrix = pd.DataFrame(index=states, columns=states)
    for state in states:
        matrix.ix[state, state] = 0
        for other in borders[state]:
            matrix.ix[state, other] = matrix.ix[other, state] = 1
    assert pd.isnull(matrix.ix["ME", "CA"])  # should not have been filled out yet
    for i in range(2, 100):  # should never get anywhere near 100
        if i > 90:
            raise RuntimeError("we have a problem! distance growing too large")
        print(i)
        number_was_used = False
        for state in states:
            most_recently_reached = [x for x in states if matrix.ix[state, x] == i - 1]
            for x in most_recently_reached:
                for y in borders[x]:
                    if pd.isnull(matrix.ix[state, y]):
                        matrix.ix[state, y] = matrix.ix[y, state] = i
                        number_was_used = True
        if not number_was_used:
            # every other cell is unreachable by any path in the graph, so make it a larger distance (i.e., the one that we just failed to use)
            for x in states:
                for y in states:
                    if pd.isnull(matrix.ix[x, y]):
                        matrix.ix[x, y] = matrix.ix[y, x] = i
            break

    return matrix


def plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap as Basemap
    from matplotlib.colors import rgb2hex
    from matplotlib.patches import Polygon
    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    # draw state boundaries.
    # data from U.S Census Bureau
    # http://www.census.gov/geo/www/cob/st2000.html
    shp_info = m.readshapefile('st99_d00','states',drawbounds=True)
    # population density by state from
    # http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
    popdensity = {
    'New Jersey':  438.00,
    'Rhode Island':   387.35,
    'Massachusetts':   312.68,
    'Connecticut':    271.40,
    'Maryland':   209.23,
    'New York':    155.18,
    'Delaware':    154.87,
    'Florida':     114.43,
    'Ohio':  107.05,
    'Pennsylvania':  105.80,
    'Illinois':    86.27,
    'California':  83.85,
    'Hawaii':  72.83,
    'Virginia':    69.03,
    'Michigan':    67.55,
    'Indiana':    65.46,
    'North Carolina':  63.80,
    'Georgia':     54.59,
    'Tennessee':   53.29,
    'New Hampshire':   53.20,
    'South Carolina':  51.45,
    'Louisiana':   39.61,
    'Kentucky':   39.28,
    'Wisconsin':  38.13,
    'Washington':  34.20,
    'Alabama':     33.84,
    'Missouri':    31.36,
    'Texas':   30.75,
    'West Virginia':   29.00,
    'Vermont':     25.41,
    'Minnesota':  23.86,
    'Mississippi':   23.42,
    'Iowa':  20.22,
    'Arkansas':    19.82,
    'Oklahoma':    19.40,
    'Arizona':     17.43,
    'Colorado':    16.01,
    'Maine':  15.95,
    'Oregon':  13.76,
    'Kansas':  12.69,
    'Utah':  10.50,
    'Nebraska':    8.60,
    'Nevada':  7.03,
    'Idaho':   6.04,
    'New Mexico':  5.79,
    'South Dakota':  3.84,
    'North Dakota':  3.59,
    'Montana':     2.39,
    'Wyoming':      1.96,
    'Alaska':     0.42}
    # choose a color for each state based on population density.
    colors={}
    statenames=[]
    cmap = plt.cm.hot # use 'hot' colormap
    vmin = 0; vmax = 450 # set range.
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            pop = popdensity[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            colors[statename] = cmap(1.-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
        statenames.append(statename)
    # cycle through state names, color each one.
    ax = plt.gca() # get current axes instance
    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
            color = rgb2hex(colors[statenames[nshape]]) 
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)
    plt.title('Filling State Polygons by Population Density')
    plt.show()


if __name__ == "__main__":
    # m = get_distance_matrix()
    # pd.DataFrame.to_csv(m, "StateDistanceMatrix.csv")

    plot()
