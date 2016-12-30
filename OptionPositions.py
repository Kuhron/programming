import random

strikes = [1950 + i for i in range(0,105,5)]
position_range = [i for i in range(-200,201)]

def make_position():
    return {"calls":{i:random.choice(position_range) if i>=2000 else 0 for i in strikes},
            "puts":{i:random.choice(position_range) if i<= 2000 else 0 for i in strikes}
    }

def pad(n):
	if n == 0:
		return " "*4
	a = str(n)
	return (" "*(4-len(a)))+a

def show_position(position):
	for x in strikes:
		print(pad(position["calls"][x])+" | "+pad(x)+" | "+pad(position["puts"][x]))

def option_position(position):
	return sum(position["calls"].values()) + sum(position["puts"].values())

def option_position_at_strike(x,position):
	return position["calls"][x] + position["puts"][x]

def skew_position(position):
	# do not include ATM options
	return sum([option_position_at_strike(strikes[i],position) for i in range(0,strikes.index(2000))])-\
	    sum([option_position_at_strike(strikes[i],position) for i in range(strikes.index(2000)+1,len(strikes))])

p = make_position()
show_position(p)
print()

# just for me to practice evaluating the position myself, rather than worrying about getting it exactly right, so don't bother checking input
waste = input("What is the net option position? ")
print(option_position(p))
waste = input("What is the net skew position? ")
print(skew_position(p))







