import math, random

all_colors = ["red","orange","yellow","green","blue","purple"]
all_interests = ["math","computers","animals","politics","sports","space","fashion","travel","music","movies","games","poetry","books","history","partying",
    "religion","nature"]

def loop_diff(lst,a,b):
    return min([abs(j*len(lst)+lst.index(a)-lst.index(b)) for j in [-1,0,1]])

# while True:
#     aa = input()
#     if aa == "":
#         break
#     a = aa.split(" ")
#     print(loop_diff(all_colors,a[0],a[1]))

def get_age():
    return random.random()**2*52+18 # distributes from 18 to 70, with older ages less common

class P: #person, ai personality with own attributes, preferences, and memories
    def __init__(self, name):
        #self.self = self
        self.name = name
        self.favorite_color = random.choice(all_colors)
        self.favorite_color_pref = random.random()
        self.age = get_age()
        self.age_pref = random.random()
        self.interests = random.sample(all_interests,random.randint(int(len(all_interests)/4),int(len(all_interests)/2)))
        self.interests_pref = random.random()
        self.memory = {}

    def describe(self):
        print("{0} is a {1:.0f}-year-old. Their favorite color is {2}, and they are interested in {3}."\
            .format(self.name,self.age,self.favorite_color,", ".join(sorted(self.interests))))

    def common_interests(self,P2):
        return set.intersection(set(self.interests),set(P2.interests))

    def assess(self, P2):
        # every term should be weight times one of the following:
        # a) a measurement of how ALIKE the people are
        # b) reciprocal of a measurement of how DIFFERENT the people are (plus one to avoid zero division)
        return self.favorite_color_pref/(1+loop_diff(all_colors,self.favorite_color,P2.favorite_color)) \
        + self.age_pref/(1+abs(self.age-P2.age)) \
        + self.interests_pref*len(self.common_interests(P2))

    def conversate(self):
        return random.choice(self.interests)

    def recall(self, P2, t):
        if P2 not in self.memory:
            return 0
        if t not in self.memory[P2]:
            return 1
        return 2

    def print_recall(self, P2, t):
        a = self.recall(P2, t)
        if a == 0:
            print("{0} has no memory of {1} at all.".format(self.name, P2.name))
        elif a == 1:
            print("{0} has no memory of what {1} was doing at time {2}.".format(self.name, P2.name, t))
        elif a == 2:
            print("{0} remembers that {1} was talking about {2} at time {3}.".format(self.name, P2.name, self.memory[P2][t], t))
        else:
            print("P.print_recall: invalid code received from P.recall")

    # def likes(self,P2,verbose = False):
    #     r = 100.0/self.assess_difference(P2)
    #     if verbose:
    #         print("{0} likes {1} {2:.1f}%.".format(self.name,P2.name,r))
    #     return r

theomax = 1.0/(1+0) + 1.0/(1+abs(0)) + 1.0*(int(len(all_interests)/2))
# print("Theoretical maximum assessment: %.2f" % theomax)

# g = []
# for i in range(10**5):
#     p = P("")
#     q = P("")
#     g.append(p.assess(q))
#     g.append(q.assess(p))
# import matplotlib.pyplot as plt
# plt.hist(g, range=(0,theomax), bins=10*theomax)#, log=True)
# plt.show()

def will_remember_event(P1,P2):
    return random.random() < P1.assess(P2)/theomax

Alice = P("Alice")
Bob = P("Bob")
Carly = P("Carly")
David = P("David")
Ella = P("Ella")
Frank = P("Frank")
Gina = P("Gina")
Hal = P("Hal")

people = [Alice,Bob,Carly,David,Ella,Frank,Gina,Hal][:8]
times = range(3)

for t in times:
    conversations = {}
    for p in people:
        conversations[p] = p.conversate()
    for p in people:
        for r in people:
            if will_remember_event(p,r):
                if r not in p.memory:
                    p.memory[r] = {t:conversations[r]}
                elif t not in p.memory[r]:
                    p.memory[r][t] = conversations[r]
                else:
                    print("Why does {0} already have a memory of {1} at time {2}?".format(p.name,r.name,t))

for p in people:
    for r in people:
        if r in p.memory:
            for t in p.memory[r]:
                a = p.recall(r,t)
                if a == 2:
                    p.print_recall(r,t)
        else:
            p.print_recall(r,0)
print()

print("Self-evaluations:")
for p in people:
    print(p.name,":",p.assess(p))
print()

print("Evaluation Matrix: (row's evaluation of column)")
print(" ".join([" "*2]+[" "*3+people[u].name[0]+" "*3 for u in range(len(people))]))
for y in people:
    print(" ".join([y.name[0]+" "]+[" {0:.3f} ".format(y.assess(x)) for x in people]))









