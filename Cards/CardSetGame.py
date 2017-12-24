import random
import matplotlib.pyplot as plt

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
    a = input("restrictions ")
    b = input("bid ")

    rc = "rb"
    rn0 = 2
    rn1 = 14
    rs = "cdhs"

    for i in a.split():
        if i in "rb":
            rc = i
        elif "-" in i:
            rn0 = int(i.split("-")[0])
            rn1 = int(i.split("-")[1])
        elif i in "cdhs":
            rs = i

    if b != "":
        b = int(b)

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
            b = int(b)
            while b <= t:
                b = input("bid ")
                b = int(b)
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