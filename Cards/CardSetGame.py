# 2018-11-05
# I don't remember how this game works, and the code was written hastily on an airplane so it doesn't explain much.
# need to refactor to use Card class but more importantly, need to give more instructive prompts to the user.
# Once figure out how it works, add a message that is printed at the beginning to explain the rules (if user wants to see it).


import random
import matplotlib.pyplot as plt

rule_str = """
You will take turns bidding how many points you are willing to bet that a card
with the given restrictions will come up. You specify the restrictions each
round yourself, and the computer simply bids what it thinks that is worth.
In a game with humans, the restriction declarer would rotate, and the bid would
start to their left.

If you get the bid, i.e., all other players pass, you will see "It's yours.",
otherwise the computer will declare "It's mine."

The highest bidder will receive 100 points if a card matching the restrictions
comes up, otherwise they will lose their bid. In this program, your score is
the difference between you and the computer. In a game with other humans, it is
probably easier to keep score for each individual and compare them, rather than
altering scores for those who passed the bid.
\n\n
"""
print(rule_str)

class Card:
    def __init__(self,string):
        self.string = string
        self.v = string[0]
        self.n = int(self.v) if self.v in "23456789" else [10,11,12,13,14]["tjkqa".index(self.v)]
        self.s = string[1]
        self.c = "r" if self.s in "dh" else "b" if self.s in "cs" else None

cs = [Card(v+s) for v in "23456789tjkqa" for s in "cdhs"]
d = cs[:]
x = 0
l = []
z = 51/1*100
j = random.choice([True,False])
k = 0
while k < 1000: # while d != []:
    changed = False
    while not changed:
        a = input("Choose restrictions, e.g., 's' for any spade, '4-6 r' for a red 4 5 or 6, '11-14' for a J Q K or A.\n")
    
        rc = "rb"
        rn0 = 2
        rn1 = 14
        rs = "cdhs"
    
        for i in a.split():
            if i in "rb":
                rc = i
                changed = True
            elif "-" in i:
                rn0 = int(i.split("-")[0])
                rn1 = int(i.split("-")[1])
                changed = rn0 != 2 or rn1 != 14
            elif i in "cdhs":
                rs = i
                changed = True
        if not changed:
            print("no restrictions made! try again")

    print("chosen restrictions: color = {}, suit = {}, {} <= number <= {}".format(rc, rs, rn0, rn1))

    b = None
    while b is None:
        b_inp = input("bid (press enter to pass)\n")
        if b_inp != "":
            try:
                b = int(b_inp)
            except ValueError:
                print("Invalid int. Try again.")

    p = len([i for i in filter(lambda y: y.s in rs and y.c in rc and y.n >= rn0 and y.n <= rn1, d)])
    q = len(d) - p
    f = 100*float(p)/q if q != 0 else z

    t = int(f*random.choice(range(80,121))/100.0)
    o = None
    while b < f-1:
        if b >= t:
            t = int(b+(f-b)*random.choice(range(80,int(10000/80.0)))/100.0)
        print("i bid",t)
        b = input("bid ")
        if b != "":
            try:
                b = int(b)
            except ValueError:
                print("invalid int")
                continue
            while b <= t:
                b_inp = input("bid ")
                try:
                    b = int(b_inp)
                except ValueError:
                    print("invalid int")
                    continue
            if b >= f-1: # f-1 because the new t would just equal b
                print("It's yours.")
                o = True
                break
        else:
            print("It's mine.")
            b = t
            o = False
            break
    if o is None:
        print("It's yours.")
        o = True

    c = random.choice(d)
    d.remove(c)
    if d == []:
        d = cs[:]
        k += 1

    print(c.string)

    if (lambda y: y.s in rs and y.c in rc and y.n >= rn0 and y.n <= rn1)(c):
        if o:
            x += 100
            print("+100 --> %d" % x)
        else:
            x -= 100
            print("-100 --> %d" % x)
    else:
        if o:
            x -= b
            print("-%d --> %d" % (b,x))
        else:
            x += b
            print("+%d --> %d" % (b,x))
    l.append(x)
    print()

    j = not j

print(x)
plt.plot(l)
plt.show()
