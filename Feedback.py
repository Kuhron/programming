import math
import random
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

class Node:
	def __init__(self):
		a = random.uniform(-2, 2)
		damping = random.uniform(0, 1)
		omega = random.uniform(0, 0.5)
		self.initial_params = {"a": a, "damping": damping, "omega": omega}
		self.params = {"a": None, "damping": None, "omega": None}
		self.min_param_values = {"a": -2, "damping": 0, "omega": 0}
		self.max_param_values = {"a": 2, "damping": 1, "omega": 1}
		self.receivers = []
		self.history = []
		self.stored_input = random.uniform(-1, 1)
		self.value = 0
		self.position = None

	def reset(self):
		a = random.uniform(-2, 2)
		damping = random.uniform(0, 1)
		omega = random.uniform(0, 0.5)
		self.initial_params = {"a": a, "damping": damping, "omega": omega}
		# self.params = {"a": None, "damping": None, "omega": None} # don't replace sliders with None
		self.min_param_values = {"a": -2, "damping": 0, "omega": 0}
		self.max_param_values = {"a": 2, "damping": 1, "omega": 1}
		# self.receivers = []
		# self.history = []
		self.stored_input = random.uniform(-1, 1)
		self.value = 0
		# self.position = None
		for param_name in self.params:
			slider = self.params[param_name]
			slider.set(self.initial_params[param_name])


	def output(self, input):
		try:
			a = self.get_param_value("a")
			omega = self.get_param_value("omega")
			transformed_input = a * self.damp(input)
			return omega * transformed_input + (1 - omega) * self.value
		except OverflowError:
			print("input {0} caused overflow".format(input))
			raise

	def damp(self, input):
		damping = self.get_param_value("damping")
		return damping**math.log(1+abs(input)) * input
		# return abs(math.log(1+abs(input))) * math.copysign(1, input)

	def plot_damping(self):
		xs = np.arange(0, 1, 0.01)
		plt.plot(xs, [self.damp(x) for x in xs])
		plt.show()

	def store_input(self, input):
		self.stored_input += input

	def eval_input(self):
		output = self.output(self.stored_input)
		self.value = output
		self.history.append(output)
		self.stored_input = self.value
		for node in self.receivers:
			node.store_input(output)

	def set_slider(self, param_name, slider):
		self.params[param_name] = slider
		slider.set(self.initial_params[param_name])

	def get_param_value(self, param_name):
		return self.params[param_name].get()

	def get_color(self):
		cmap = matplotlib.cm.get_cmap("Spectral")
		max_value_range = (-0.25, 0.25)
		quantile = (self.value - max_value_range[0]) *1.0/ (max_value_range[1] - max_value_range[0])
		return cmap(quantile)

	def add(self, amount=None):
		val = amount if amount else self.number_box.get()
		try:
			val = float(val)
			if val in [np.nan, np.inf, -np.inf]:
				raise ValueError
			self.stored_input += val
		except ValueError:
			print("invalid value {0}".format(val))

	def set_number_box(self, number_box):
		self.number_box = number_box


class Network:
	def __init__(self, n_nodes):
		self.nodes = [Node() for i in range(n_nodes)]
		self.connect_nodes()

	def connect_nodes(self):
		for node in self.nodes:
			f = lambda _: random.choice([True, False])
			# f = lambda _: True
			receivers = [node2 for node2 in self.nodes if node2 is not node and f(node2)]
			node.receivers = receivers

	# def run(self):
	# 	# self.nodes[0].store_input(1)
	# 	for i in range(100):
	# 		for node in self.nodes:
	# 			node.eval_input()

	# def plot_history(self, fig):
	# 	fig.plot(self.nodes[0].history)
	# 	for i in range(1, len(self.nodes)):
	# 		fig.plot(self.nodes[i].history)
	# 	fig.show()

	def get_histories(self):
		return [node.history for node in self.nodes]

	def plot_sum_history(self):
		sum_history = sum([np.array(node.history) for node in self.nodes])
		plt.plot(sum_history)
		plt.show()

	def reset(self):
		for node in self.nodes:
			node.reset()
		self.connect_nodes()
		# node histories will also have been destroyed by this, as new nodes were initialized


class NodeController:
	def __init__(self, master, network):
		self.master = master
		self.network = network
		self.nodes = network.nodes
		self.setup()

	def setup(self):
		# sets up the GUI
		self.canvas_area = tk.Frame(self.master)
		self.canvas_area.pack(side=tk.LEFT)
		self.slider_area = tk.Frame(self.master)
		self.slider_area.pack(side=tk.LEFT)

		self.node_fig = plt.figure(1)
		self.history_fig = plt.figure(2)

		self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=self.canvas_area)
		self.history_canvas.show()
		self.history_canvas.get_tk_widget().config(height=200, width=400)
		self.history_canvas.get_tk_widget().pack(side=tk.BOTTOM)
		self.toolbar = NavigationToolbar2TkAgg(self.history_canvas, self.canvas_area)
		self.toolbar.update()
		# self.history_canvas._tkcanvas.pack(side=tk.BOTTOM) # redundant?

		self.node_canvas = FigureCanvasTkAgg(self.node_fig, master=self.canvas_area)
		self.node_canvas.show()
		self.node_canvas.get_tk_widget().config(height=200, width=400)
		self.node_canvas.get_tk_widget().pack(side=tk.BOTTOM)
		self.toolbar = NavigationToolbar2TkAgg(self.node_canvas, self.canvas_area)
		self.toolbar.update()
		# self.node_canvas._tkcanvas.pack(side=tk.BOTTOM)

		self.sliders = self.get_sliders()

		shock_row = tk.Frame(self.slider_area)
		shock_row.pack(side=tk.TOP)
		shock_box = tk.Entry(shock_row)
		shock_box.pack(side=tk.LEFT)
		self.shock_box=shock_box
		shock_button = tk.Button(shock_row, text="Shock", command=self.shock)
		shock_button.pack(side=tk.LEFT)

		button_row = tk.Frame(self.slider_area)
		button_row.pack(side=tk.TOP)

		quit_button = tk.Button(button_row, text='Quit', command=self.quit)
		quit_button.pack(side=tk.LEFT)
		self.master.protocol("WM_DELETE_WINDOW", self.quit)

		reset_button = tk.Button(button_row, text='Reset', command=self.reset)
		reset_button.pack(side=tk.LEFT)

		self.node_positions = self.set_node_positions()

		self.color_cycle = "rgbcmyk"

	def reset(self):
		self.network.reset()
		self.nodes = self.network.nodes
		self.node_fig.clear()

	def plot_nodes(self):
		fig = self.node_fig
		ax = fig.gca()

		for node in self.nodes:
			node.eval_input()

		ax.set_xlim(-1.5, 1.5)
		ax.set_ylim(-1.5, 1.5)
		color = lambda: random.choice("rgb")
		colors = [node.get_color() for node in self.nodes]
		ax.scatter(self.node_positions[0], self.node_positions[1], c=colors, s=300)

		for i in range(len(self.nodes)):
			color_designation = self.color_cycle[i % len(self.color_cycle)]
			node = self.nodes[i]
			r = 1.2
			ax.scatter([self.node_positions[0][i]*r], [self.node_positions[1][i]*r], c=color_designation, s=100)

		for node1 in self.nodes:
			for node2 in node1.receivers:
				r = 0.7
				x1, y1 = node1.position
				x2, y2 = node2.position
				xmid, ymid = (x1+x2)*0.5, (y1+y2)*0.5
				x1_, y1_ = xmid + (x1-xmid)*r, ymid + (y1-ymid)*r
				x2_, y2_ = xmid + (x2-xmid)*r, ymid + (y2-ymid)*r
				
				# diff = (node2.position[0]*r-node1.position[0]*r, node2.position[1]*r-node1.position[1]*r)
				diff = (x2_-x1_, y2_-y1_)
				ax.arrow(x1_, y1_, diff[0], diff[1], head_width=0.05)

		self.node_canvas.draw()
		self.master.after(10, self.plot_nodes)

	def plot_histories(self):
		fig = self.history_fig
		fig.clear()
		ax = fig.gca()

		n_points = 10

		ax._get_lines.set_color_cycle(self.color_cycle)

		hists = self.network.get_histories()
		for hist in hists:
			ax.plot(hist[-n_points:])

		self.history_canvas.draw()
		self.master.after(10, self.plot_histories)

	def quit(self):
	    self.master.quit()
	    self.master.destroy()

	def get_sliders(self):
		# for help with how to think about layouts: http://zetcode.com/gui/tkinter/layout/
		# currently creating sliders and placing as they are made because I can't change their parents later on
		result = []
		# slider_index = 0
		rows = [tk.Frame(self.slider_area) for i in range(len(self.nodes))]
		for i in range(len(self.nodes)):
			node = self.nodes[i]
			row = rows[i]
			row.pack(side=tk.TOP, fill=tk.X)
			for param_name in sorted(node.params.keys()):
				min_value = node.min_param_values[param_name]
				max_value = node.max_param_values[param_name]
				slider = tk.Scale(row, from_=min_value, to=max_value, resolution=(max_value-min_value)*0.01,
					orient=tk.HORIZONTAL,
					label=param_name if i == 0 else None
					)
				slider.pack(side=tk.LEFT)
				node.set_slider(param_name, slider)

				result.append(slider)

			number_box = tk.Entry(row, width = 5)
			number_box.pack(side=tk.LEFT)
			node.set_number_box(number_box)

			add_button = tk.Button(row, text="+", command=node.add)
			add_button.pack(side=tk.LEFT)
		# for slider in self.sliders:
		# 	row_index = slider_index // sliders_per_row
		# 	slider.pack(side=tk.LEFT, anchor=tk.NW)
			# slider_index += 1
		return result

	def set_node_positions(self):
		n = len(self.nodes)
		angles = np.arange(0, 2*np.pi, (2*np.pi)*1.0/n)
		xs = [math.sin(angle) for angle in angles]
		ys = [math.cos(angle) for angle in angles]
		# return [i for i in zip(xs, ys)]
		
		for i in range(n):
			self.nodes[i].position = (xs[i], ys[i])
		return [xs, ys]

	def run(self):
		# call all the functions that will be repeated
		# within these functions you will need to reschedule the task by having a similar self.master.after statement that calls the same function
		self.plot_nodes()
		self.plot_histories()

		# now start it running
		self.master.mainloop()
		plt.close()
		# while True:
		# 	self.plot_nodes()
		# 	time.sleep(0.1)

	def shock(self):
		max_shock = float(self.shock_box.get())
		for i in range(len(self.nodes)):
			self.nodes[i].add(random.uniform(-max_shock, max_shock))



net = Network(7)

controller = NodeController(tk.Tk(), net)
controller.run()

# net.run()
# net.plot_history()
# net.plot_sum_history()
# net.nodes[0].plot_damping()














