import random

def get_main_numbers():
	return sorted(random.sample(range(1,70),5))

def get_powerball_number():
	return random.sample(range(1,27),1)

def get_numbers():
	return [get_main_numbers(), get_powerball_number()]

def get_lst_matches(lst1, lst2):
	return len(set(lst1) & set(lst2))

def get_matches(numbers1, numbers2):
	return tuple([get_lst_matches(numbers1[0],numbers2[0]), get_lst_matches(numbers1[1],numbers2[1])])

def is_jackpot(matches):
	return matches == (5,1)

def get_prize(matches, prizes):
	matches = tuple(matches)
	return prizes[matches] if matches in prizes else 0

def get_average_payoff(prizes, n_trials):
	total_payoff = 0
	for i in range(int(n_trials)):
		matches = get_matches(get_numbers(), get_numbers())
		if is_jackpot(matches):
			print("jackpot!")
		total_payoff += get_prize(matches, prizes)
	return total_payoff * 1.0/n_trials

jackpot = 1.3e9
prizes = {
	(0,1):4,
	(1,1):4,
	(2,1):7,
	(3,0):7,
	(3,1):100,
	(4,0):100,
	(4,1):5e4,
	(5,0):1e6,
	(5,1):jackpot
}

# print(get_average_payoff(prizes, 1e6))
print(get_numbers())