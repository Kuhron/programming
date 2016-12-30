import math
import random

import numpy as np
import matplotlib.pyplot as plt


# start by just simulating agents with different utility functions for food and water, and they die if they don't have enough
# see what properties become common after many generations

class Params:
	def __init__(self, dct=None):
		self.dct = dct if dct else {
			"discount_factor": random.random(),
			"food_consumption": random.random(),
			"water_consumption": random.random(),
			"asexual_reproduction_rate": random.uniform(0, 0.1),
			"sexual_reproduction_rate": random.uniform(0, 0.01),
			# "energy_consumption": random.random(),
			"energy_from_food": random.random(),
			"energy_from_storage": -1*random.random(),
			"energy_from_sexual_reproduction": -1*random.random(),
			"energy_from_asexual_reproduction": -1*random.random()
		}
		self.bounds = {
			"discount_factor": (0, 1),
			"food_consumption": (0.1, np.inf),
			"water_consumption": (0.1, np.inf),
			"asexual_reproduction_rate": (0, 1),
			"sexual_reproduction_rate": (0, 1),
			# "energy_consumption": (0.25, np.inf),
			"energy_from_food": (0, np.inf),
			"energy_from_storage": (-np.inf, -0.3),
			"energy_from_sexual_reproduction": (-np.inf, -0.3),
			"energy_from_asexual_reproduction": (-np.inf, -0.3)
		}
		self.bound()

	def get(self, param_name):
		return self.dct.get(param_name)

	def get_max_time_horizon(self):
		try:
			return int(math.log(1e-6, self.dct["discount_factor"])) # x | self.discount_factor ** x == 1e-6
		except ZeroDivisionError:
			print("#DIV/0! with discount factor of {0}".format(self.dct["discount_factor"]))
			raise

	def bound(self):
		result = {}
		for k in self.dct.keys():
			v = self.dct[k]
			min_, max_ = self.bounds[k]
			result[k] = max(min_, min(max_, v))
		self.dct = result

	def offset(self):
		result = {}
		for k in self.dct.keys():
			sign = math.copysign(1, self.dct[k])
			val = abs(self.dct[k])
			v = sign * math.exp(math.log(val) * random.uniform(0.5, 1.5))
			min_, max_ = self.bounds[k]
			result[k] = max(min_, min(max_, v))
		return Params(result)

	def __add__(self, other):
		if type(other) in [Params]:
			dct = {k: self.dct[k] + other.dct[k] for k in self.dct.keys()}
		else:
			return NotImplemented
		return Params(dct)

	def __mul__(self, other):
		if type(other) in [int, float]:
			dct = {k: self.dct[k] * other for k in self.dct.keys()}
		elif type(other) in [dict]:
			dct = {k: self.dct[k] * (other[k] if k in other else 1) for k in self.dct.keys()}
		elif type(other) in [Params]:
			dct = {k: self.dct[k] * (other.dct[k] if k in other.dct else 1) for k in self.dct.keys()}
		else:
			return NotImplemented
		return Params(dct)

	def __rmul__(self, other):
		return self.__mul__(other)


class Stockpiles:
	def __init__(self, dct=None):
		self.dct = dct if dct else {
			"water": random.random(),
			"food": random.random(),
			"energy": random.random()
		}

	def get(self, resource):
		return self.dct.get(resource)

	def __add__(self, other):
		if type(other) in [dict]:
			dct = {k: self.dct[k] + (other[k] if k in other else 0) for k in self.dct.keys()}
		elif type(other) in [Stockpiles]:
			dct = {k: self.dct[k] + (other.dct[k] if k in other.dct else 0) for k in self.dct.keys()}
		else:
			return NotImplemented
		return Stockpiles(dct)

	def __iter__(self):
		for k in self.dct.keys():
			yield k


class Agent:
	def __init__(self, params=None, stockpiles=None):
		self.params = params if params else Params()
		self.stockpiles = stockpiles if stockpiles else Stockpiles()
		self.alive = True

	def reproduce_asexually(self):
		energy_consumed = self.params.get("energy_from_asexual_reproduction")
		if self.stockpiles.get("energy") + energy_consumed >= 0:
			self.stockpiles += {"energy": energy_consumed}
			return Agent(self.params.offset())
		else:
			return None

	def reproduce_sexually(self, other_agent):
		energy_consumed = self.params.get("energy_from_sexual_reproduction")
		if self.stockpiles.get("energy") + energy_consumed >= 0:
			self.stockpiles += {"energy": energy_consumed}
			alpha = random.random()
			new_params = alpha * self.params.offset() + (1 - alpha) * other_agent.params.offset()
			return Agent(new_params)
		else:
			return None

	def reproduce(self, all_agents):
		result = []
		if random.random() < self.params.get("asexual_reproduction_rate"):
			a = self.reproduce_asexually()
			if a:
				result.append(a)
		f = lambda _: random.random() < self.params.get("sexual_reproduction_rate")
		agents_to_reproduce_with = [i for i in filter(lambda x: x is not self and f(x), all_agents)]
		for agent in agents_to_reproduce_with:
			a = self.reproduce_sexually(agent)
			if a:
				result.append(a)
		return result

	def store(self, resource):
		if resource in ["energy"]:
			return
		anticipated_need = sum([(self.params.get("discount_factor")**t) * self.params.get(resource + "_consumption")
			for t in range(self.params.get_max_time_horizon())])
		current_storage = self.stockpiles.get(resource)
		amount_to_store = (anticipated_need - current_storage) * random.uniform(0, 1)
		energy_consumed = self.params.get("energy_from_storage") * amount_to_store
		self.stockpiles += {resource: amount_to_store, "energy": energy_consumed}

	def consume(self, resource):
		if resource in ["energy"]:
			return
		need = self.params.get(resource + "_consumption") * random.uniform(0.5, 1.5)
		consumption = min(need, self.stockpiles.get(resource))
		need -= consumption
		self.stockpiles += {resource: -1*consumption}
		if resource == "food":
			energy_consumed = consumption * self.params.get("energy_from_food") # will be positive unlike all the other energy consumptions
			self.stockpiles += {"energy": energy_consumed}
		if need > 1e-6:
			self.die()

	def pass_day(self, all_agents):
		if self.is_alive():
			for resource in self.stockpiles:
				self.store(resource)
			for resource in self.stockpiles:
				self.consume(resource)
			new_agents = self.reproduce(all_agents)
			return new_agents
		else:
			return []

	def is_alive(self):
		return self.alive

	def die(self):
		self.alive = False


agents = [Agent() for i in range(100)]
param_names = agents[0].params.dct.keys()
t = 0
min_t = 20
max_t = 1e5
max_agents = 500
percentile_values = [0, 25, 50, 75, 100]
quantiles = {k: [] for k in param_names}
populations = []
# for t in range(100):
while t < min_t or (len(agents) < max_agents and t < max_t):
	print("time {0}, {1} agents".format(t, len(agents)))
	new_agents = []
	for agent in agents:
		new_agents.extend(agent.pass_day(agents))
	agents.extend(new_agents)
	agents = [i for i in filter(lambda x: x.is_alive(), agents)]
	agents = random.sample(agents, min(max_agents, len(agents)))
	if agents == []:
		agents = [Agent()]
	if len(agents) > 1:
		populations.append(len(agents))
		for param_name in param_names:
			vals = [agent.params.get(param_name) for agent in agents if agent.is_alive()]
			quantiles[param_name].append(tuple([np.percentile(vals, q) for q in percentile_values]))
	t += 1

for param_name in param_names:
	# plt.hist([agent.params.get(param_name) for agent in agents if agent.is_alive()])
	tuples = quantiles[param_name]
	for i in range(len(percentile_values)):
		plt.plot([tu[i] for tu in tuples])
	plt.title(param_name)
	plt.show()

plt.plot(populations)
plt.title("population")
plt.show()

