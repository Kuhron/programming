import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def printr(n):
    n = str(int(n))
    n = " "*(8-len(n))+n
    print(n, end = "\r")

# note that int(-0.5) = 0; all negatives truncate UP
# test to prove this
#q = []
#for i in range(10**6):
    #q.append(int(random.normalvariate(0,5)))
#print(sum(q)/len(q))

def random_walk_display():
    a = 0
    while True:
        a += int(random.normalvariate(0,5))
        printr(a)

def random_walk_graph():
    a = 0
    while True:
        q = random_walk_list(a)
        plt.plot(range(len(q)),q)
        a = q[-1]
        plt.show()

def random_walk_list(start, length):
    a = start
    result = []
    for j in range(length):
        a += int(random.normalvariate(0,5))
        result.append(a)
    return result

#random_walk_graph()

# from http://matplotlib.org/1.4.2/examples/animation/simple_anim.html
w = 100

def pad_g(g):
    return [0 for i in range(w-len(g))]+g

global global_walk_a, global_walk_b, gg
global_walk_a = random_walk_list(0,100000)
global_walk_b = random_walk_list(0,100000)
gg = [global_walk_a[i] - global_walk_b[i] for i in range(100000)]

plt.plot(gg)
plt.show()
import sys
sys.exit()

fig, ax = plt.subplots()

x = np.arange(0, 100, 1)        # x-array
line, = ax.plot(x, gg[:w])
plt.ylim(-100,100)

def animate(i):
    global gg
    if i <= 1:
        g = pad_g([0])
    elif i <= w:
        g = pad_g(gg[:i])
    else:
        g = gg[i-w:i]
    #if len(global_walk) < 10:
        #global_walk += random_walk_list(global_walk[-1],1000)
    line.set_ydata(g)  # update the data
    #plt.axis([i,w+i,min(-100,min(g)),max(100,max(g))])
    #plt.xlim(i,w+i)
    plt.ylim(min(-100,min(g)),max(100,max(g)))
    printr(g[-1])
    return line,

#Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = anim.FuncAnimation(fig, animate, init_func=init,
    interval=1, blit=True)
plt.show()
