import pyglet
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# https://groups.google.com/g/pyglet-users/c/NnF2xH_5GSY?pli=1

MIN_PRESSURE_FOR_STROKE = 0.001


def get_tablets(display=None):
    # Each cursor appears as a separate xinput device; find devices that look
    # like Wacom tablet cursors and amalgamate them into a single tablet. 
    valid_names = ('stylus', 'cursor', 'eraser', 'wacom', 'pen', 'pad')
    cursors = []
    devices = get_devices(display)
    for device in devices:
        dev_name = device.name.lower().split()
        if any(n in dev_name for n in valid_names) and len(device.axes) >= 3:
            cursors.append(XInputTabletCursor(device))
    if cursors:
        return [XInputTablet(cursors)]
    return []


def get_array_from_data_fp(fp, binarize_pressure_threshold):
    with open(fp) as f:
        lines = f.readlines()
    while "" in lines:
        lines.remove("")
    assert lines[0].startswith("time_ms")  # header row
    l = [[float(x) for x in line.strip().split("\t")] for line in lines[1:]]
    l = np.array(l)
    assert l.shape[-1] == 4
    xs = l[:, 1]
    ys = l[:, 2]
    pressures_raw = l[:, 3]

    if binarize_pressure_threshold is None:
        # don't change pressures
        pass
    else:
        if ((pressures_raw == 0) | (pressures_raw == 1)).all():
            pass  # they're already binary
        else:
            pressures_binary = pressures_raw >= binarize_pressure_threshold
            l[:, 3] = pressures_binary
    return l


def plot_xyp_time_series(l, show=True):
    assert type(l) is np.ndarray
    n_time_points, n_channels = l.shape
    if n_channels == 4:
        ts = l[:, 0]
        xs = l[:, 1]
        ys = l[:, 2]
        ps = l[:, 3]
    elif n_channels == 3:
        # no time specified
        ts = list(range(l.shape[0]))
        xs = l[:, 0]
        ys = l[:, 1]
        ps = l[:, 2]
    else:
        raise ValueError(f"bad shape {l.shape}")

    # time series of x, y, and pressure values
    plt.scatter(ts, xs, label="x", c="r")
    plt.scatter(ts, ys, label="y", c="b")
    plt.scatter(ts, ps, label="p", c="y")
    plt.legend()
    if show:
        plt.show()


def draw_glyph_from_xyp_time_series(l, pressure_threshold, show=True):
    # draw the shape that the data describes
    # I think the NN can completely ignore time? it just has a series of x,y values (and whether the pen lifts)
    assert type(l) is np.ndarray
    strokes = []
    current_stroke = []
    n_channels = l.shape[-1]
    pressures = l[:, -1]
    if not ((pressures == 0) | (pressures == 1)).all():
        pressures_binary = pressures >= pressure_threshold
        l[:, -1] = pressures_binary
    for i in range(len(l)):
        if n_channels == 4:
            t,x,y,p = l[i]
        elif n_channels == 3:
            x,y,p = l[i]
        else:
            raise ValueError(f"bad shape: {l.shape}")
        current_stroke.append([x, y])
        # IMPORTANT: now we assume logistic pressure so the NN can output a logit of whether the pen is pressed down
        if p < 0.5 or i == len(l) - 1:
            # end of this stroke
            strokes.append(current_stroke)
            current_stroke = []

        # if i == len(l) - 1:
        #     assert p == 0, l[i]  # not necessarily true for NN-generated ones
    for stroke in strokes:
        xs = [xy[0] for xy in stroke]
        ys = [xy[1] for xy in stroke]
        plt.plot(xs, ys, c="k")
    plt.axis("equal")
    if show:
        plt.show()


def plot_simultaneous_strokes(l, show=True):
    n_strokes, n_time_points, n_channels = l.shape
    assert n_channels == 2
    n_channels = 2
    for i in range(n_strokes):
        plt.subplot(int(round(n_strokes/n_channels)), n_channels, i+1)
        xs = l[i, :, 0]
        ys = l[i, :, 1]
        plt.plot(xs)
        plt.plot(ys)
    if show:
        plt.show()


def draw_glyph_from_simultaneous_strokes(l, show=True):
    n_strokes, n_time_points, n_channels = l.shape
    assert n_channels == 2
    for i in range(n_strokes):
        xs = l[i, :, 0]
        ys = l[i, :, 1]
        plt.plot(xs, ys, c="k")
    plt.axis("equal")
    if show:
        plt.show()


def write_data_to_file_from_xyp_time_series(arr, output_fp):
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    n_time_points, n_channels = arr.shape
    with open(output_fp, "w") as f:
        f.write("time_ms\tx_01\ty_01\tpressure_01\n")
        for i in range(n_time_points):
            if n_channels == 4:
                t,x,y,p = arr[i]
            elif n_channels == 3:
                t = i
                x,y,p = arr[i]
            else:
                raise ValueError(f"bad shape {arr.shape}")
            f.write(f"{t}\t{x}\t{y}\t{p}\n")


def write_data_to_file_from_simultaneous_strokes(arr, output_fp):
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    n_strokes, n_time_points, n_channels = arr.shape
    assert n_channels == 2  # x and y
    headers = []
    for i in range(n_strokes):
        headers += [f"x{i}", f"y{i}"]
    header = "\t".join(headers) + "\n"
    with open(output_fp, "w") as f:
        f.write(header)
        for i in range(n_time_points):
            row = []
            for j in range(n_strokes):
                x = arr[j, i, 0]
                y = arr[j, i, 1]
                row += [x, y]
            row_str = "\t".join(str(x) for x in row) + "\n"
            f.write(row_str)


if __name__ == "__main__":
    collect_data = True
    plot_data = True

    window_x = 540
    window_y = 540
    window = pyglet.window.Window(window_x, window_y)
    batch = pyglet.graphics.Batch()

    tablet = pyglet.input.get_tablets()[0]
    tablet_canvas = tablet.open(window)
    t0 = time.time()
    motion_data = []
    last_touch_point = None
    last_x_01, last_y_01 = 0, 0
    lines = []  # so the line object won't be garbage collected before the batch can draw it on the window


    @tablet_canvas.event
    def on_motion(cursor_name, x, y, pressure, *extra_args):
        assert all(a == 0 for a in extra_args), extra_args
        t_ms = (time.time() - t0) * 1000
        x_01 = x / window_x
        y_01 = y / window_y

        global last_touch_point  # seems like a bad idea but whatever, people are doing it
        if last_touch_point is not None or pressure >= MIN_PRESSURE_FOR_STROKE:
            # don't keep track of all the non-touching events
            motion_data.append([t_ms, x_01, y_01, pressure])  # just record actual pressure here, don't binarize it, do that later when prepping training data, this is just raw data recording

        print(f"t = {int(t_ms)} ms, x = {x_01:.6f}, y = {y_01:.6f}, {pressure = :.6f}")

        if pressure < MIN_PRESSURE_FOR_STROKE:
            last_touch_point = None
        else:
            if last_touch_point is not None:
                # draw most recent line segment
                x1, y1 = last_touch_point
                x2, y2 = x, y
                line = pyglet.shapes.Line(x1, y1, x2, y2, color=(255, 255, 0), batch=batch)
                global lines
                lines.append(line)
            last_touch_point = (x, y)


    @window.event
    def on_draw():
        # gets called constantly throughout the running of the app
        window.clear()
        batch.draw()


    @window.event
    def on_key_press(symbol, modifiers):
        global motion_data
        if symbol == pyglet.window.key.SPACE or symbol == pyglet.window.key.ENTER:
            if symbol == pyglet.window.key.ENTER:
                # write data to file for NN training
                now_str = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S")
                output_fname = f"{now_str}.tsv"
                output_fp = os.path.join("VineScriptTabletInputData", output_fname)
                write_data_to_file(motion_data, output_fp)
            # clear the window and lines, reset time
            global t0
            t0 = time.time()
            global lines
            lines = []
            motion_data = []
            window.clear()


    if collect_data:
        window.clear()
        pyglet.app.run()

    if plot_data:
        subdir = None
        if subdir is None:
            data_fps = sorted([os.path.join("VineScriptTabletInputData", x) for x in os.listdir("VineScriptTabletInputData") if x.endswith(".tsv")])
        else:
            data_fps = sorted([os.path.join("VineScriptTabletInputData", subdir, x) for x in os.listdir(os.path.join("VineScriptTabletInputData", subdir)) if x.endswith(".tsv")])
        for fp in data_fps:
            l = get_array_from_data_fp(fp, binarize_pressure_threshold=None)
            print(fp)
            # plot_xyp_time_series(l)
            draw_glyph_from_xyp_time_series(l, pressure_threshold=MIN_PRESSURE_FOR_STROKE)


