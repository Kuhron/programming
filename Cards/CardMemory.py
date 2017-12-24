import random

import matplotlib.pyplot as plt
import numpy as np

from Card import Card, DeckOfCards


class PerfectPlayer:
	def __init__(self):
		self.card_memory = {value: 0 for value in Card.VALUES}

	def add_card(self, card):
		self.card_memory[card.value] += 1

	def guess(self, verbose=False):
		min_count = min(self.card_memory.values())
		candidates = [value for value, count in self.card_memory.items() if count == min_count]
		value = random.choice(candidates)
		if verbose:
			print("I guess", value)
		return Card(value, "S")


class RandomPlayer:
	def __init__(self):
		pass

	def add_card(self, card):
		pass

	def guess(self, verbose=False):
		value = random.choice(Card.VALUES)
		if verbose:
			print("I guess", value)
		return Card(value, "S")


class HumanPlayer:
	def __init__(self):
		pass

	def add_card(self, card):
		pass

	def guess(self, verbose=False):
		value = input("Guess card value: ")
		return Card(value, "S")


def get_physical_card():
	value, suit = input("Card drawn: ")
	card = Card(value, suit)
	return card


def get_input_series(using_physical_deck=False):
	if using_physical_deck:
		for i in range(52):
			yield get_physical_card  # call the function when want to know the card
	else:
		deck = DeckOfCards()
		deck.shuffle()
		for card in deck.deal():
			yield (lambda: card)


def play(n_times, using_physical_deck=False):
	players = [PerfectPlayer() for i in range(1)]
	players.append(HumanPlayer())
	high_score = 0
	all_scores = []
	for t in range(n_times):
		scores = np.array([0 for player in players])
		series = get_input_series(using_physical_deck)
		for card_holder in series:
			print("\nguessing next card")
			guesses = np.array([player.guess(verbose=True).value for player in players])
			card = card_holder()
			scores += (guesses == card.value)
			print(card)
			print("scores: {0} (mean: {1})".format(scores if len(players) <= 10 else "[...]", np.mean(scores)))
			for player in players:
				player.add_card(card)
		all_scores = list(all_scores) + list(scores)
		high_score = max(high_score, max(scores))
		print("high score:", high_score)
	print("done")
	# print("mean score: ", np.mean(all_scores))

	# min_score = min(all_scores)
	# max_score = max(all_scores)
	# bins = np.arange(min_score - 0.5, max_score + 1.5, 1)
	# plt.hist(all_scores, bins)
	# plt.show()


play(1, using_physical_deck=True)