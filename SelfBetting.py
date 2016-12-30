import math
import pytz

from datetime import datetime, timedelta, timezone

import numpy as np
import matplotlib.pyplot as plt


# how to use:
# in the csv, place a bet, with BetMadeUTC being when it goes into effect, and BetClosedUTC being when it is deactivated
# TotalStakes is the total dollar amount, where it will all be lost if the event happens either immediately or at infinity
# the payoff is determined by some metric such that total win of the stakes occurs when the event happens at the expected time
# zero payoff is therefore somewhere in between, depending on the metric
# use dollars as a function of probability that the event would have occurred by time t, under poisson dist given your log lambda
# so 0% and 100% lead to total loss because you were wrong, and 50% gives total gain
# ideally the payoff is something "spikier" than linear, so there is more incentive to be accurate
# extreme would be a dirac delta function of probability, where you only win if it happens exactly at your expected time (PayoffDegree = 0)
# lower PayoffDegree corresponds to spikier function, with lower integral, so accuracy is more crucial (PayoffDegree \in R+)

# to change a bet, close out the old one and make a new one. the entire stakes of the old bet will be lost
# use the honor system on this, so you aren't gaming it by changing lambda to make maximum payoff always be now or something like that


class Bet:
	def __init__(self, description, bet_made_utc, bet_closed_utc, first_occurrence_utc, ln_lambda_hz, total_stakes, payoff_degree, row_number):
		self.description = description
		self.bet_made_utc = datetime.strptime(bet_made_utc + " +0000", "%Y-%m-%d-%H:%M:%S %z")
		self.bet_closed_utc = datetime.strptime(bet_closed_utc + " +0000", "%Y-%m-%d-%H:%M:%S %z") if bet_closed_utc != "None" else None
		self.first_occurrence_utc = datetime.strptime(first_occurrence_utc + " +0000", "%Y-%m-%d-%H:%M:%S %z") if first_occurrence_utc != "None" else None
		self.ln_lambda_hz = float(ln_lambda_hz)
		self.total_stakes = float(total_stakes)
		self.payoff_degree = float(payoff_degree)
		if self.payoff_degree < 0:
			raise ValueError("Bet has payoff_degree < 0. Description: {0}".format(self.description))
		self.row_number = int(row_number)

	def metric(self, p):
		d = self.payoff_degree
		return 1 if p == 0.5 and d == 0 else 1-(2**(d+1))*(abs(p - 0.5))**d

	def payoff(self, p):
		return self.total_stakes * self.metric(p)

	def graph_metric(self):
		probs = np.arange(0,1,0.001)
		ys = [self.metric(p) for p in probs]
		plt.plot(probs, ys)
		plt.title(self.description)
		plt.xlabel("probability that event occurs by this time")
		plt.ylabel("payoff metric (normalized)")
		plt.show()
		plt.close()

	def time_to_probability(self, dt_utc):
		t0 = self.bet_made_utc
		td = dt_utc - t0
		t = td.total_seconds()
		lam = math.exp(self.ln_lambda_hz)
		p_no_occurrence = poisson_prob(n=0, dt=t, lam=lam)
		return 1 - p_no_occurrence

	def graph_probability_by_time(self, max_horizon):
		td = max_horizon
		x_min = self.bet_made_utc
		x_max = min(self.bet_closed_utc, x_min + td) if self.bet_closed_utc is not None else x_min + td
		xs = [x_min + i*td for i in np.arange(0,1,0.001)]
		ys = [self.time_to_probability(x) for x in xs]
		plt.plot(xs, ys)
		plt.title(self.description)
		plt.xlabel("datetime")
		plt.ylabel("probability that event occurs by this time")
		labels = plt.gca().get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		# now = datetime.now(timezone.utc)
		# if now >= x_min and now <= x_max:
		# 	pass# plt.vertical_line(now)

		plt.show()
		plt.close()

	def payoff_by_time(self, t):
		return self.payoff(self.time_to_probability(t))

	def graph_payoff_by_time(self, max_horizon):
		td = max_horizon
		x_min = self.bet_made_utc
		x_max = min(self.bet_closed_utc, x_min + td) if self.bet_closed_utc is not None else x_min + td
		xs = [x_min + i*td for i in np.arange(0,1,0.001)]
		ys = [self.payoff_by_time(x) for x in xs]
		plt.plot(xs, ys)
		plt.title(self.description)
		plt.xlabel("datetime")
		plt.ylabel("payoff if event occurs at this time ($)")
		labels = plt.gca().get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)
		
		# now = datetime.now(timezone.utc)
		# if now >= x_min and now <= x_max:
		# 	pass# plt.vertical_line(now)

		plt.show()
		plt.close()

	def is_open(self):
		now = datetime.now(timezone.utc)
		return self.bet_closed_utc is None or now >= self.bet_made_utc and now < bet_closed_utc

	def total_loss(self):
		return -1*self.total_stakes

	def final_payoff(self):
		if self.first_occurrence_utc is None:
			return self.total_loss()

		if self.bet_closed_utc is None or (self.first_occurrence_utc >= self.bet_made_utc and self.first_occurrence_utc < self.bet_closed_utc):
			return self.payoff(self.time_to_probability(self.first_occurrence_utc))
		else:
			return self.total_loss()


def poisson_prob(n, dt, lam):
	return 1.0/math.factorial(n) * (lam * dt)**n * math.exp(-lam * dt)

def read_bets():
	f = open("Self Betting Record.csv", "r")
	lines = f.readlines()
	f.close()
	lines = [i.strip().split(",") for i in lines]
	header = lines[0]
	print("header:", header)
	bets = {}
	for i in range(1, len(lines)):
		line = lines[i]
		row_number = i + 1
		bets[row_number] = create_bet_from_line(line, header, row_number)
		#for k in range(len(header)):
			#bets[row_number][header[k]] = create_bet_from_line(line[k])
	return bets

def create_bet_from_line(line, header, row_number):
	args = [line[i] for i in range(len(header))]
	args.append(row_number)
	return Bet(*args)


bets = read_bets()
for row_number in bets.keys():
	bet = bets[row_number]
	print("%35s" % bet.description, "%5.2f" % bet.payoff_by_time(pytz.utc.localize(datetime.now())))
	# bet.graph_probability_by_time(timedelta(seconds=86400*365*2))
	# bet.graph_payoff_by_time(timedelta(seconds=86400*365*2))


