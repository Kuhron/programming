# modeling my strategy of using time before and during flights to get my body on destination time
# see what patterns of sleep/wake times would work best for a given time zone offset

# initial simplifications:
# - only one flight, nonstop
# - (maybe?) ignore when the flight itself is (since in reality the passenger has to be awake at departure and arrival times to do airport stuff)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def make_2d_timeline_plot_bare():
    # x axis is time
    # y axis varies the offset between source and destination time zones
    # I decided it will be nicer and symmetrical to have both source and destination isochrons be diagonal for varying y, rather than one of them being used to hold a given local time at a constant x value

    # show at least 48 hours, probably better to have 72, to get the full transition in the plot for any offset
    x0 = -36
    x1 = 36
    plt.xlim((x0, x1))
    plt.xticks(range(x0, x1+6, 6))

    # also show some redundant y values so I can see how there's (hopefully) a continuous gradation of sleep-wake plans along the offsets
    y0 = -18
    y1 = 18
    plt.ylim((y0, y1))
    plt.yticks(range(y0, y1+3, 3))

    # color-code the isochrons by time of day
    color_12am = "black"
    color_6am = "yellow"
    color_12pm = "orange"
    color_6pm = "red"

    # at offset zero, put the middle of the x axis at noon local
    
    # making it easier on myself, just scatter some time dots so I see what it looks like, figure out equations of stuff from that once I have the visual

    
    src_dots_12pm = [(0, 0), (-24, 0), (24, 0), (3, 6)]
    dest_dots_12pm = [(0, 0), (-24, 0), (24, 0)]
    src_dots_6pm = [(6, 0), (-18, 0), (30, 0)]
    dest_dots_6pm = [(6, 0), (-18, 0), (30, 0), (3, 6)]

    xs_12pm, ys_12pm = zip(*(src_dots_12pm + dest_dots_12pm))
    xs_6pm, ys_6pm = zip(*(src_dots_6pm + dest_dots_6pm))

    plt.scatter(xs_12pm, ys_12pm, c=color_12pm, alpha=0.5)
    plt.scatter(xs_6pm, ys_6pm, c=color_6pm, alpha=0.5)
    plt.gca().set_aspect("equal")

    src_lines_12pm = [((-y1/2, -y1), (y1/2, y1))]
    dest_lines_12pm = []
    lines_12pm = src_lines_12pm + dest_lines_12pm

    lc_12pm = LineCollection(lines_12pm, color=color_12pm)
    plt.gca().add_collection(lc_12pm)


if __name__ == "__main__":
    sfo_to_sin_offset = ((+8) - (-7)) # in summer
    sfo_to_sin_flight_time = 17

    make_2d_timeline_plot_bare()
    plt.savefig("JetLagPlot.png")

