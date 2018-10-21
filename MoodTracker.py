# original note called "Morning Mood" on my phone
# collect data on drift over time
# using scale to 7

from datetime import datetime
import matplotlib.pyplot as plt

# original scale (note asymmetry where 4 is better than neutral)
# 1 the worst
# 2 bad, mediocre
# 3 meh
# 4 alright, a bit good but still mild
# 5 good
# 6 very good, strong feeling
# 7 the best, very rare

# symmetrical scale
scale = {
    1: "the worst, rare",
    2: "very bad",
    3: "bad",
    4: "neutral",
    5: "good",
    6: "very good",
    7: "the best, rare",
}

# conversion old -> new = 1:1 2:3 3:4 4:4.5 5:5 6:6 7:7
# there is old data in the note from the fall of 2016, which I may not use

# 2016-10-18 3
# 2016-10-19 2
# 2016-10-20 4
# 2016-10-21 4
# 2016-10-28 4
# 2016-10-31 3
# 2016-11-01 2
# 2016-11-02 2
# 2016-11-03 3
# 2016-11-04 4
# 2016-11-21 1
# 2016-11-22 2
# 2016-11-23 2

data_fp = "MoodTrackerData.csv"
dt_format = "%Y-%m-%d %H:%M:%S UTC"


def record():
    numbers = scale.keys()
    print("Mood right now?")
    for n in numbers:
        print("{} : {}".format(n, scale[n]))
    print("or type \"skip\" to just plot existing data")
    while True:
        inp = input()
        if inp == "skip":
            res = None
            break
        try:
            res = int(inp.strip())
        except ValueError:
            print("try again")
            continue
        if res not in numbers:
            print("try again")
            continue
        break

    dt = datetime.utcnow()
    dt_str = dt.strftime(dt_format)

    if res is not None:
        with open(data_fp, "a") as f:
            f.write("{},{}\n".format(dt_str, res))


def get_plot_data():
    with open(data_fp) as f:
        lines = [line.strip() for line in f.readlines()]
    assert lines[0] == "dt,mood"
    lines = [line.split(",") for line in lines[1:]]
    xs = [datetime.strptime(line[0], dt_format) for line in lines]
    ys = [int(line[1]) for line in lines]
    return xs, ys

def plot():
    xs, ys = get_plot_data()
    plot_raw_scatter(xs, ys, "Raw Scatter")
    plot_time_of_year(xs, ys)  # do later if have data over lots of time
    plot_time_of_day(xs, ys)

def plot_raw_scatter(xs, ys, title):
    x_range = max(xs) - min(xs)
    padding = x_range / 24
    plt.scatter(xs, ys)
    plt.xlim(xmin = min(xs) - padding, xmax = max(xs) + padding)
    plt.title(title)
    plt.show()

def plot_time_of_year(xs, ys):
    xs = [x.replace(year=2000) for x in xs]
    plot_raw_scatter(xs, ys, "Seasonality")

def plot_time_of_day(xs, ys):
    xs = [x.replace(year=2000, month=1, day=1) for x in xs]
    plot_raw_scatter(xs, ys, "Time of Day")


if __name__ == "__main__":
    record()
    plot()
