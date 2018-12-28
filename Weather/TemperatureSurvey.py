import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from datetime import datetime, timedelta


data_fp = "Temperature Preferences Responses - Form Responses 1.csv"

df = pd.read_csv(data_fp)

# some cleaning
df = df[df["data_mistake"].isnull()]  # quick and dirty but there are false positives that need to be put back in
# need to switch the temperature scales for the people who made mistakes there
# don't remove the response from Saskatchewan (who still lives there) that has really low preferences, because it is more believable for that one to be Celsius (as reported) than Fahrenheit; thus I would bet that it is not a mistake
df = df[(df["lowF"] <= df["optimumF"]) & (df["optimumF"] <= df["highF"])]

all_temps = set(list(df["lowF"]) + list(df["optimumF"]) + list(df["highF"]))
minx = min(all_temps)
maxx = max(all_temps)
rangex = maxx - minx

def p_hists():
    plt.title("low, opt, and high temps in deg F")
    dtemp_in_bin = 2  # deg F
    # round min and max bin bounds such that each bin is centered on a multiple of dtemp_in_bin
    minbin = int(dtemp_in_bin * (-0.5 + int(minx / dtemp_in_bin)))
    maxbin = int(dtemp_in_bin * (1.5 + int(maxx / dtemp_in_bin)))  # hist bins sequence requires both endpoints
    bins = range(minbin, maxbin, dtemp_in_bin)
    plt.subplot(3, 1, 1)
    plt.hist(df["lowF"], color="blue", bins=bins)
    plt.xlim(minx, maxx)
    plt.subplot(3, 1, 2)
    plt.hist(df["optimumF"], color="green", bins=bins)
    plt.xlim(minx, maxx)
    plt.subplot(3, 1, 3)
    plt.hist(df["highF"], color="red", bins=bins)
    plt.xlim(minx, maxx)
    plt.show()
# p_hists()

def p_cumdists():
    lows = sorted(df["lowF"])
    opts = sorted(df["optimumF"])
    highs = sorted(df["highF"])
    plt.plot(lows, color="blue")
    plt.plot(opts, color="green")
    plt.plot(highs, color="red")
    plt.show()
p_cumdists()

def p_responses():
    plt.title("individual responses")
    # show all responses in some order, like rows on the plot, with each row having the low, opt, high points
    # sort by low, or opt, or something
    # df2 = df.sort_values(["optimumF", "lowF", "highF"])
    # df2 = df.sort_values(["optimumF"])
    # df2 = df.sort_values(["lowF"])
    # df2 = df.sort_values(["highF"])
    df2 = df.sort_values(["optimumF", "rangeF"])

    df2 = df2.reset_index()
    # print(df2["optimumF"])
    line_segments = []
    colors = []
    for index, row in df2.iterrows():
        y = index
        # for column, color in zip(["lowF", "optimumF", "highF"], ["blue", "green", "red"]):
        #     plt.scatter(row[column], y, color=color)
        x0 = row["lowF"]
        x1 = row["optimumF"]
        x2 = row["highF"]
        line_segments.append([(x0, y), (x1, y)])
        colors.append("blue")
        line_segments.append([(x1, y), (x2, y)])
        colors.append("red")
    lc = mc.LineCollection(line_segments, colors=colors, linewidths=1)
    plt.gca().add_collection(lc)
    plt.gca().autoscale()
    plt.show()
p_responses()

def p_acceptability():
    plt.title("how many people each temperature is acceptable to")
    is_too_cold = lambda x: (x < df["lowF"]).sum()
    is_acceptable = lambda x: ((df["lowF"] <= x) & (x <= df["highF"])).sum()
    is_too_hot = lambda x: (df["highF"] < x).sum()
    xs = range(int(minx), int(maxx) + 1, 1)
    plt.plot(xs, [is_too_cold(x) for x in xs], color="blue", label="too cold")
    plt.plot(xs, [is_acceptable(x) for x in xs], color="green", label="acceptable")
    plt.plot(xs, [is_too_hot(x) for x in xs], color="red", label="too hot")
    plt.legend()
    plt.show()
p_acceptability()

def p_time():
    plt.title("responses over time")
    times = sorted(datetime.strptime(x, "%m/%d/%Y %H:%M:%S") for x in df["Timestamp"])
    f = lambda t: sum(t2 <= t for t2 in times)
    mint = min(times)
    maxt = max(times)
    td = timedelta(hours=1)
    current_t = mint.replace(minute=0, second=0)
    ts = []
    while current_t < maxt + td:
        ts.append(current_t)
        current_t += td
    ys = [f(t) for t in ts]
    plt.plot(ts, ys)
    plt.show()
# p_time()

def p_range():
    plt.title("range in deg F")
    plt.hist(df["rangeF"], bins=100)
    plt.show()
p_range()

def p_opt01():
    # TODO this graph's x-axis is messed up
    plt.title("optimum in interval [0, 1]")
    plt.hist(df["opt01"], bins=100)
    # plt.xticks(np.arange(0, 1.1, 0.1))
    plt.show()
p_opt01()
