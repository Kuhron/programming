import math
import msvcrt
import random
import time

import matplotlib.pyplot as plt
import numpy as np


class Event:
	def __init__(self, p):
		self.p = p

	def trial(self):
		return random.random() < self.p


class DeltaBet:
	def __init__(self, delta):
		self.delta = delta

	def payoff(self, outcome):
		# the payoff based on (delta) and (1 - delta) is supposed to consolidate a bunch of guesses on your part
		# e.g. the outcome is True and your delta is 0.8. Then you would guess correctly 80% of the time and wrongly 20% of the time.

		correct_coefficient = 1.0
		incorrect_coefficient = -2.0
		guess = random.random() < self.delta
		# return 1 if guess == outcome else -1 # induces just guessing all True or all False (or flipping a coin if delta=0.5)
		correct_delta = self.delta if outcome else (1 - self.delta)
		incorrect_delta = (1 - self.delta) if outcome else self.delta
		# return 1.0/correct_delta if guess == outcome else -1.0/incorrect_delta # EV zero for all delta, with higher variance on wings
		# return correct_delta - incorrect_delta # induces guessing all True or all False

		correct_payoff = float(correct_coefficient)/correct_delta
		incorrect_payoff = float(incorrect_coefficient)/incorrect_delta
		return correct_payoff + incorrect_payoff #+ abs(incorrect_coefficient *1.0/ correct_coefficient) # normalize optimum to 0


class LogDeltaBet:
	def __init__(self, log_delta):
		self = DeltaBet(10**log_delta)


def test_payoff_function():
	e = Event(0.2)
	# deltas = np.arange(0, 1, 0.01)
	deltas = np.arange(0.01, 1.00, 0.01)
	n_trials = 10**5
	total_payoffs = []
	for delta in deltas:
		bet = DeltaBet(delta)
		total_payoff = 0
		for i in range(n_trials):
			total_payoff += bet.payoff(e.trial())
		total_payoffs.append(total_payoff)

	average_payoffs = [i *1.0/ n_trials for i in total_payoffs]
	plt.plot(deltas, average_payoffs)
	plt.show()

def get_best_right_ratio(p):
	r_star = p**2 + (1-p)**2
	return r_star

def get_deviance_from_best_right_ratio(right_ratio, p, normalize):
	r_star = get_best_right_ratio(p)
	q = r_star if normalize else 1
	return ((r_star - right_ratio) *1.0/ q)**2

def get_score(right_ratio, p, normalize):
	deviance = get_deviance_from_best_right_ratio(right_ratio, p, normalize)
	return -1*math.log(deviance) if deviance != 0 else np.nan


# pricing of delta bets
# 
# payoff should be maximized when delta = p, and monotonically decrease for greater error
# payoff for a given trial should not depend on what p actually is, so no one has to know it
# being right about more improbable events should pay off more (inversely gives constant expected value)
# symmetry between delta that event occurs and (1 - delta) that it does not

# pricing of lambda bets
# 
# class LambdaBet:
# class LogLambdaBet:

def simulate_manual_delta_bet_1(event):
	delay = 0.001 # seconds
	n_trials = 1
	p = event.p
	p_hat = None
	trues = 0
	falses = 0
	rights = 0
	wrongs = 0
	rs = []
	r_stars = []
	p_hats = []
	s1s = []
	s2s = []
	while True:
		try:
			guess = False
			t = time.time()
			print("Trial {0}. Press enter to quit. Press any key if you bet that the outcome will be True in the next {1} seconds.".format(
				n_trials, delay))

			while time.time() < t + delay:
				if msvcrt.kbhit():
					if ord(msvcrt.getch()) == 3:
						raise KeyboardInterrupt
					guess = True

			outcome = event.trial()
			if outcome:
				trues += 1
				if guess:
					rights += 1
				else:
					wrongs += 1
			else:
				falses += 1
				if guess:
					wrongs += 1
				else:
					rights += 1

			p_hat = trues *1.0/ (trues + falses)
			r = rights *1.0/ (rights + wrongs)
			r_star = get_best_right_ratio(p_hat)
			s1 = get_score(r, p_hat, normalize=False)
			s2 = get_score(r, p_hat, normalize=True)
			rs.append(r)
			r_stars.append(r_star)
			p_hats.append(p_hat)
			s1s.append(s1)
			s2s.append(s2)
			print("outcome {0:5s}, guess {1:5s}, right {2:4d}, wrong {3:4d}, score {4:2.2f}, normscore {5:2.2f}".format(
				str(outcome), str(guess), rights, wrongs, s1, s2))
			n_trials += 1
		except KeyboardInterrupt:
			print("exiting")
			break

	plt.plot(rs, label="right ratio")
	plt.plot(r_stars, label="ideal right ratio")
	plt.plot(p_hats, label="p estimate")
	ax = plt.gca()
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
	plt.show()

	plt.plot(s1s, label="score")
	plt.plot(s2s, label="normscore")
	ax = plt.gca()
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
	plt.show()


if __name__ == "__main__":
	action = input("1. delta bet\n2.lambda bet\n")
	if action == "1":
		p = random.uniform(0, 1)
		e = Event(p)
		# b = DeltaBet()
		action2 = input("1. manual choice\n2. specify delta\n")
		if action2 == "1":
			# b.score_manually()
			simulate_manual_delta_bet_1(e)
		elif action2 == "2":
			# blah
			raise lambda: lambda: lambda: lambda: lambda: lambda: lambda: None
	elif action == "2":
		print(NotImplemented)
	else:
		raise Exception("invalid action")
















