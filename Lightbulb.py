import random

class Prisoner:
    def __init__(self,index):
        self.index = index
        self.is_counter = self.index == 0
        self.has_flipped_switch = self.is_counter

inspection_mode = input("Would you like to inspect the output as it occurs? (y/n; default = no)") == "y"

def report(s):
	if inspection_mode:
		print(s)
	else:
		pass

def wait_for_user():
	if inspection_mode:
		waste = input()
	else:
		pass

N = 100
count = 0
n_trials = 0
lightbulb_on = False
prisoners = [Prisoner(i) for i in range(N)]

while count < N-1:
	p = random.choice(prisoners)
	report("\nTrial %d" % n_trials)
	report("prisoner %d enters" % p.index)
	if lightbulb_on:
		report("lightbulb is on")
		if p.is_counter:
			report("prisoner is the counter, sees the lightbulb is on, and turns it off")
			# the counter does not bother counting themself because for the count to advance at all, they must see the lightbulb
			count += 1
			lightbulb_on = False
		else:
			report("prisoner is not the counter and does not do anything")
			# do nothing
			pass
	else:
		report("lightbulb is off")
		if not p.has_flipped_switch:
			report("prisoner is here for the first time and turns the lightbulb on")
			lightbulb_on = True
			p.has_flipped_switch = True
	n_trials += 1
	wait_for_user()

print("Complete. Took %d trials." % n_trials)