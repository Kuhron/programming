import inspect
import math
import random
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg


class GraphController:
    def __init__(self, master, func_dict, resolution):
        self.master = master
        self.func_dict = func_dict
        self.resolution = resolution
        assert len(set(get_arg_names(f) for f in func_dict.values())) == 1, "functions must all have same kwargs (for sliders)"
        self.arg_names = get_arg_names(next(iter(self.func_dict.values())))
        self.active_variable = "a"
        self.variable_arrays = {arg_name: list(np.arange(-1, 1, 2/self.resolution)) for arg_name in self.arg_names}
        self.arg_arrays = {arg_name: np.nan for arg_name in self.arg_names}

        self.setup()

    def setup(self):
        # sets up the GUI
        self.canvas_area = tk.Frame(self.master)
        self.canvas_area.pack(side=tk.LEFT)
        self.slider_area = tk.Frame(self.master)
        self.slider_area.pack(side=tk.LEFT)

        self.fig = plt.figure(1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_area)
        self.canvas.show()
        self.canvas.get_tk_widget().config(height=500, width=500)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.canvas_area)
        self.toolbar.update()

        self.func_string_var = tk.StringVar()
        self.func_string_var.set(next(iter(self.func_dict.keys())))
        option_menu = tk.OptionMenu(self.slider_area, self.func_string_var, *self.func_dict.keys())
        option_menu.pack(side=tk.TOP)

        self.create_sliders()

        # shock_row = tk.Frame(self.slider_area)
        # shock_row.pack(side=tk.TOP)
        # shock_box = tk.Entry(shock_row)
        # shock_box.pack(side=tk.LEFT)
        # self.shock_box=shock_box
        # shock_button = tk.Button(shock_row, text="Shock", command=self.shock)
        # shock_button.pack(side=tk.LEFT)

        button_row = tk.Frame(self.slider_area)
        button_row.pack(side=tk.TOP)

        quit_button = tk.Button(button_row, text='Quit', command=self.quit)
        quit_button.pack(side=tk.LEFT)
        self.master.protocol("WM_DELETE_WINDOW", self.quit)

        reset_button = tk.Button(button_row, text='Reset', command=self.reset)
        reset_button.pack(side=tk.LEFT)

    def reset(self):
        self.fig.clear()

    def activate(self, arg_name):
        self.active_variable = arg_name

    def get_func(self):
        return self.func_dict[self.func_string_var.get()]

    def update_arg_arrays(self):
        for arg_name, slider in self.sliders.items():
            new_val = slider.get()
            self.arg_arrays[arg_name] = [new_val]

        self.arg_arrays[self.active_variable] = self.variable_arrays[self.active_variable]

    def update_slider_bounds(self):
        for arg_name in self.arg_names:
            try:
                min_bound = float(self.min_boxes[arg_name].get())
                max_bound = float(self.max_boxes[arg_name].get())
            except ValueError:
                continue
            slider = self.sliders[arg_name]
            slider.configure(from_=min_bound, to=max_bound)
            self.variable_arrays[arg_name] = np.arange(min_bound, max_bound, (max_bound - min_bound) / self.resolution)

    def plot(self):
        fig = self.fig
        fig.clear()
        ax = fig.gca()

        n_points = 10

        self.update_slider_bounds()
        self.update_arg_arrays()
        self.ys = get_function_values(self.get_func(), self.arg_arrays)
        ax.plot(self.variable_arrays[self.active_variable], self.ys)

        self.canvas.draw()
        self.master.after(10, self.plot)

    def quit(self):
        self.master.quit()
        self.master.destroy()

    def create_sliders(self):
        # for help with how to think about layouts: http://zetcode.com/gui/tkinter/layout/

        self.sliders = {}
        self.min_boxes = {}
        self.max_boxes = {}

        rows = [tk.Frame(self.slider_area) for i in range(len(self.arg_names))]
        for i in range(len(self.arg_names)):
            arg_name = self.arg_names[i]
            row = rows[i]
            row.pack(side=tk.TOP, fill=tk.X)

            min_number_box = tk.Entry(row, width = 5)
            min_number_box.pack(side=tk.LEFT)
            self.min_boxes[arg_name] = min_number_box

            min_value = -1  # default
            max_value = 1  # default

            slider = tk.Scale(row, from_=min_value, to=max_value, resolution=(max_value-min_value)*0.01,
                orient=tk.HORIZONTAL,
                label=arg_name
            )
            slider.pack(side=tk.LEFT)
            slider.set((min_value + max_value) / 2)
            self.sliders[arg_name] = slider

            max_number_box = tk.Entry(row, width = 5)
            max_number_box.pack(side=tk.LEFT)
            self.max_boxes[arg_name] = max_number_box

            callback = lambda arg_name=arg_name: self.activate(arg_name)  # default args are evaluated on function creation, so closure is retained
            add_button = tk.Button(row, text="activate", command=callback)
            add_button.pack(side=tk.LEFT)


    def run(self):
        # call all the functions that will be repeated
        # within these functions you will need to reschedule the task by having a similar self.master.after statement that calls the same function
        self.plot()

        self.master.mainloop()
        plt.close()


def get_function_values(func, arg_dict):
    lens = list(map(len, arg_dict.values()))
    assert lens.count(1) == len(lens) - 1 and max(lens) > 1, "please provide a list of lists of values for the positional args, with all but one of length 1"

    max_len = max(lens)
    map_dict = {}
    for arg_name, arg_array in arg_dict.items():
        if len(arg_array) == max_len:
            map_dict[arg_name] = arg_array
        else:
            map_dict[arg_name] = list(arg_array) * max_len
    # map_array = [array if len(array) == max_len else list(array) * max_len for array in arg_array]

    map_dict_list = [{arg_name: arg_array[i] for arg_name, arg_array in map_dict.items()} for i in range(max_len)]

    return [func(**d) for d in map_dict_list]  # map does not take kwargs


def get_arg_names(func):
    argspec = inspect.getfullargspec(func)
    assert len(argspec.args) == 0, "function should only take keyword-only args"
    return tuple(sorted(argspec.kwonlyargs))



func_main = lambda *, a, b, c, d: np.sin(a*b + c - a*d**2) + b*np.cos(c*a**2) * c*np.sin(a*d**2)
func_sub_1 = lambda *, a, b, c, d: np.cos(b**2 + c*d) + a*np.sin(a + b + c + d)

func_dict = {
    "f": func_main,
    "f1": func_sub_1,
}
resolution = 500

controller = GraphController(tk.Tk(), func_dict, resolution)
controller.run()




