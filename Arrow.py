# simulate motion of particles in a grid, with toroidal array

import random
import sys

# def stop(message):
#     print(message)
#     sys.exit()

# n = 8
# n_particles = 16
# t = 100

# def get_x(n,n_particles):
#     g = [[0 for i in range(n)] for j in range(n)]
#     used = []
#     for i in range(n_particles):
#         x1,x2 = (random.choice(range(n)),random.choice(range(n)))
#         while (x1,x2) in used:
#             x1,x2 = (random.choice(range(n)),random.choice(range(n)))
#         used.append((x1,x2))
#     for i in used:
#         g[x1][x2] = 1
#     return g,used

# g,x = get_x(n,n_particles)

# for t_ in range(t):
#     p_i = random.choice(range(len(x)))
#     p = x[p_i]
#     if random.random()<0.5:
#         continue
#     d = random.choice(range(4))
#     if d == 0:
#         spot = ((p[0]+1) % n,p[1])
#     elif d == 1:
#         spot = ((p[0]-1) % n,p[1])
#     elif d == 2:
#         spot = (p[0],(p[1]+1) % n)
#     elif d == 3:
#         spot = (p[0],(p[1]-1) % n)
#     if g[spot[0]][spot[1]] == 0:
#         g[spot[0]][spot[1]] = 1
#         g[p[0]][p[1]] = 0
#         x[p_i] = spot


# just some syntax if i'm adding motion vectors to coordinates
# tuple([a[i] + b[i] for i in range(len(a))]

########

# different approach here; using graph theory representation of system
# each possible state of the system is treated as a vertex, and you can travel between them based on the adjacency matrix

# representing the sparse matrix of microstate interactions as just a bunch of pairs for which vertex (microstate) i and j are connected
# keep in mind that any movement must be reversible (probability > 0), but it can be as probable or improbable as you want
# use set of tuples where (i,j) has i < j, and no state maps to itself (we're already in that state, so who cares about trying to "move there")

def show_sorted_dict(d):
    for k in sorted(d):
        print(k,":",d[k],end=", ")
    print()

class Graph:
    def __init__(self,n_states=None,A=None,M=None):
        self.n_states = n_states
        self.A = A
        self.M = M
        if M != None:
            if len(M) != len(M[0]):
                raise IndexError("M must be square matrix.")
            if M != [[M[c][r] for c in range(len(M))] for r in range(len(M))]:
                raise ValueError("M must be symmetric matrix.")

        if A == None and M == None:
            self.A = set()
            self.all_points = set([i for i in range(self.n_states)])
            self.connected_points = {0}
            self.disconnected_points = set([i for i in range(1,self.n_states)])
            while self.disconnected_points != set():
                p1 = random.choice(list(self.connected_points)) # start with 0 always to seed the graph
                p2 = random.choice(list(self.all_points - {p1})) # ensure p1 != p2
                # pair = (p1,p2) if p1 < p2 else (p2,p1)
                self.A.add((p1,p2))
                self.A.add((p2,p1))
                self.connected_points.add(p2)
                self.disconnected_points -= {p1,p2} # p1 should never be there, but just in case
        elif A != None:
            self.A = A
        if self.A != None:
            self.n_states = max(max([i[0] for i in self.A]),max([i[1] for i in self.A])) + 1
            self.M = [[(1 if (r,c) in self.A or (c,r) in self.A else 0) for c in range(self.n_states)] for r in range(self.n_states)]
        else:
            if M == None and n_states == None:
                raise ValueError("At least one of n_states, A, and M must be specified.")
            self.M = M
            self.n_states = len(self.M)
            self.A = set()
            for r in range(self.n_states):
                for c in range(r,self.n_states): # start after the diagonal and go right
                    if M[r][c] == 1:
                        self.A.add((r,c))
                        self.A.add((c,r))
            # if n_states != len(M): # this should probably go somewhere else; it was intended for M or A to supersede n_states if user makes mistake
            #     self.n_states = len(M)

        self.mem_a_set = {}
        for p in range(self.n_states):
            self.mem_a_set[p] = frozenset(self.a_set(p))

        self.mem_a_len = {}
        for p in range(self.n_states):
            self.mem_a_len[p] = len(self.mem_a_set[p])

        self.mem_h_set = {}
        for p in range(self.n_states):
            self.mem_h_set[p] = {}
            for n in range(9):
                # just initialize with trivial ones, add others only as needed within the instance methods
                self.mem_h_set[p][n] = frozenset(self.h_set(p,n))

        self.mem_h_len = {}
        for p in range(self.n_states):
            self.mem_h_len[p] = {}
            for n in range(9):
                self.mem_h_len[p][n] = len(self.mem_h_set[p][n])

        self.mem_H = {}
        for n in range(9):
            self.mem_H[n] = self.H(n)

    def describe(self):
        print("n_states:",self.n_states,end="\n"*2)
        print("A:",self.A,end="\n"*2)
        self.show_M(); print()
        print("adjacency lengths:"); show_sorted_dict(self.mem_a_len); print()
        print("history lengths:"); show_sorted_dict(self.mem_h_len);print()

    def show_M(self):
        print("\n".join([" ".join([("1" if i==1 else "-") for i in row]) for row in self.M]))

    def a_set(self,p):
        # adjacency set of p

        # get memoized answer if exist
        if p in self.mem_a_set:
            return self.mem_a_set[p]
        result = set([i for i in filter(lambda x: ((x,p) if x<p else (p,x)) in self.A,range(self.n_states))])
        self.mem_a_set[p] = result
        return result

    def a_len(self,p):
        # number of states adjacent to p

        # get memoized answer if exist
        if p in self.mem_a_len:
            return self.mem_a_len[p]

        result = len(self.a_set(p))
        
        # memoize result
        self.mem_a_len[p] = result

        return result

    def h_set(self,p,n,recursion_safety=True):
        # set of histories leading to/from p, as ordered paths

        # protect against excessive RAM usage
        if n > 8 and recursion_safety:
            # this has filled my RAM before
            raise OverflowError("Path lengths are too long (n={0}). If you want to try again, please pass recursion_safety=False to this function, " \
            "and watch your RAM.".format(n))

        # get memoized answer if exist
        if p in self.mem_h_set:
            if n in self.mem_h_set[p]:
                return self.mem_h_set[p][n]

        if n == 0:
            return set()
        if n == 1: # eta(p) = {[q] | q \in alpha(p)}
            return set([(i,) for i in self.a_set(p)])

        result = set()
        for q in self.a_set(p):
            for h in self.h_set(q,n-1):
                result.add((q,) + h)

        # memoize result
        if p in self.mem_h_set:
            self.mem_h_set[p][n] = result
        else:
            self.mem_h_set[p] = {n:result}

        return result

    def h_len(self,p,n):
        # number of histories leading to/from p

        # get memoized answer if exist
        if p in self.mem_h_len:
            if n in self.mem_h_len[p]:
                return self.mem_h_len[p][n]

        result = len(self.h_set(p,n)) # actual definition, but loses time because h_len(p,n-1) etc. are not memoized
        # return self.a_len(p)+sum([self.h_len(q,n-1) for q in self.a_set(p)]) # THIS IS WRONG

        # memoize result
        if p in self.mem_h_len:
            self.mem_h_len[p][n] = result
        else:
            self.mem_h_len[p] = {n:result}

        return result

    def H(self,n):
        # number of histories of length n in the graph

        # get memoized answer if exist
        if n in self.mem_H:
            return self.mem_H[n]

        result = sum([self.h_len(p,n) for p in range(self.n_states)])

        # memoize result
        self.mem_H[n] = result

        return result

    def r(self,p,n):
        # proportion of the histories of length n that lead to/from p
        # no need to memoize here as far as I can see
        return self.h_len(p,n)/float(self.H(n))

    def R(self,p):
        # limit as n -> inf of r(p,n,A)
        # this is the way of measuring the entropy of a state of the system (the proportion of microstates that could correspond to it)

        # find the highest tolerable value for n
        n = 50 # will probably never be close to this high
        while True:
            try:
                return(self.r(p,n))
            except OverflowError: # note: I built in an OverflowError to Graph.h_set(), so OverflowError here is a failsafe, not an actual memory failure
                n -= 1

    def Ra(self,p):
        # ratio of entropy to adjacency; this is not constant, even for a given number of adjacent states
        # the point of this measure is to see how constant it is; I noticed a positive correlation between adjacency and entropy
        return self.R(p)/float(self.a_len(p))


import matplotlib.pyplot as plt

# G = Graph(n_states=8)
G = Graph(M=[
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,1],
    [0,0,0,0,0,1,1,1],
    [0,0,0,0,1,1,1,1],
    [0,0,0,1,1,1,1,1],
    [0,0,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1]
])
# G.show_M()
# print()
# p = random.choice(range(G.n_states))
# print(p)
# xrn = [G.r(p,n) for n in range(1,9)]
# xrp = ["{0:.2f}%".format(100*G.r(p,8)) for p in range(G.n_states)]
xRp = ["{0:.2f}%".format(100*G.R(p)) for p in range(G.n_states)]
xRap = ["{0:.2f}%".format(100*G.Ra(p)) for p in range(G.n_states)]
# print(sum([float(i[:-1]) for i in xRp])) # math check: should be 100%
# plt.plot(xrn)
# plt.show()
G.describe()
print(xRp)
print(xRap)
plt.plot([G.mem_h_len[p][5] for p in range(G.n_states)])
plt.show()









