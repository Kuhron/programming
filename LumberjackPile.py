# http://www.reddit.com/r/dailyprogrammer/comments/3840rp/20150601_challenge_217_easy_lumberjack_pile/

def get_int(prompt):
    while True:
        s = input(prompt)
        try:
            s = int(s)
        except ValueError:
            print("invalid input")
            continue
        break
    return s

def get_ints(prompt):
    while True:
        s = input(prompt)
        s = s.split(" ")
        try:
            s = [int(i) for i in s]
        except ValueError:
            print("invalid input")
            continue
        break
    return s

s = get_int("How many logs long is each side of the square pile? ")
n = get_int("How many logs do we need to move onto the pile? ")

pile = []
for i in range(s):
    while True:
        x = get_ints("Please enter row number {0}, separated by spaces.\n".format(i))
        if len(x) != s:
            print("That row is the wrong length ({0} instead of {1}).".format(len(x),s))
            continue
        break
    pile.append(x)

#print(pile)

def add_logs(s,n,pile):
    while n > 0:
        pile_min = min([min(r) for r in pile])
        for i in range(s):
            for j in range(s):
                if pile[i][j] == pile_min:
                    pile[i][j] += 1
                    n -= 1
                if n <= 0: # catch weird stuff if it happens (i.e. n becomes negative somehow)
                    return pile
        pile_min += 1
    return pile # unreachable, but whatever

pile = add_logs(s,n,pile)
print()
for r in pile:
    print(" ".join([str(i) for i in r]))


