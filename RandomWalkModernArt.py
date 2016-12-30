# http://stackoverflow.com/questions/17322041/visualizing-a-2d-random-walk-in-python

# def original_walk():
#   walk = randomWalkb(25)
#   print(walk)
#   plt.plot(walk[0],walk[1],'b+', label= 'Random walk')
#   plt.axis([-10,10,-10,10])
#   plt.show()

mass = 0.9 # number between 0 and 1 that determines how likely the particle is to keep its momentum, related to expected length of straight run (=/100?)

x_max = 40
y_max = 70
border = False
if border:
    border_width = 0
    border_thick = 3
wrap_line = False
weird = False # still doesn't do what I wanted (see use in function random_walk); supposed to make the gaps between parallel lines not so quantized

def get_origin():
    return (random.randint(-x_max,x_max),random.randint(-y_max,y_max))

def random_walk(length, origin = (0,0), repeat = True, use_momentum = False):
    momentum = 0
    walk_x = [origin[0]]
    walk_y = [origin[1]]

    # also keep the steps here so that if the more efficient way of checking x before y fails, then we still can check the pair without zip
    walk_path = [origin]
    
    for i in range(length):
        try_again = True # first time for this step in the length
        stopper = 0 # allows repeating of values if an unseen one is not reached after 50 tries
        while try_again and stopper < 50:
            x,y = walk_x[-1],walk_y[-1]
            x_,y_,m_ = random_step_super(x,y,momentum=momentum)
            if use_momentum:
                momentum = m_
            if repeat or x_ not in walk_x or y_ not in walk_y:
                try_again = False
            else:
                try_again = (x_,y_) in walk_path
            stopper += 1
        if weird:
            x_ = x_ * max(random.normalvariate(1,0.5),0)
            y_ = y_ * max(random.normalvariate(1,0.5),0)
        walk_x.append(x_)
        walk_y.append(y_)
        walk_path.append((x,y))
        # if use_momentum:
        #     x_f = walk_x[-1]
        #     x_i = walk_x[-2]
        #     y_f = walk_y[-1]
        #     y_i = walk_y[-2]
        #     if x_f != x_i:
        #         momentum = iudhxueix
        #     elif y_f != y_i:
        #         momentum = uoedfxidb
        #     else:
        #         print("function random_walk: You are moving in both directions at once. Check the construction of walk_x and walk_y.")
    return walk_path
    # fucking around: return [(i/(j+1),j/(i+1)) for i in range(math.floor(math.sqrt(length))) for j in range(math.floor(math.sqrt(length)))]

def random_step_super(x,y,momentum = 0):
    # momentum equal to 0 means anything goes, can be set by having parameter use_momentum false on function random_walk
    # momentum in range(1,5) means that same number will be taken without question half the time, or a new choice will occur
    # this function does not return the momentum, as that is calculated within function random_walk from the difference in the points
    if momentum > 0 and random.random() < mass: # we have momentum and choose to use it
        x_ = x
        y_ = y
        if momentum == 1:
            x_ += 1
        elif momentum == 2:
            y_ += 1
        elif momentum == 3:
            x_ += -1
        else:
            y_ += -1
        m_ = momentum
    else: # if we have momentum, we don't use it
        x_,y_,m_ = random_step(x,y)
    x_ = wrap(x_, x_max)
    y_ = wrap(y_, y_max)
    return x_,y_,m_

def random_step(x,y):
    new = random.randint(1,4) # both ends inclusive, weird i know
    x_ = x
    y_ = y
    if new == 1:
        x_ += 1
    elif new == 2:
        y_ += 1
    elif new == 3:
        x_ += -1
    else:
        y_ += -1
    return x_,y_,new

def wrap(n, bound): # problem with toroidal array in that module turtle connects all the way across when a point is wrapped
    if n > bound:
        n -= 2*bound
    elif n < -bound:
        n += 2*bound
    return n

def main():
    import turtle

    turtle.speed("fastest")
    turtle.pen(shown = False)
    origin = get_origin()

    total_length = 4000.0 # float
    seg_length = 1000
    seg_length = int(min(total_length,seg_length))
    dilate = 3.0 # float

    for _ in range(math.ceil(total_length/seg_length)): # macro-steps
        walk = random_walk(seg_length, repeat = False, use_momentum = True, origin = origin)
        origin = walk[-1]
        for x_displace in [0]:#range(-1,2):
            for y_displace in [0]:#range(-1,2):
                turtle.pendown()
                for u in range(len(walk)):
                    x,y = walk[u][0]+(2*x_max-1)*x_displace,walk[u][1]+(2*y_max-1)*y_displace
                    if u > 0:
                        x_last,y_last = walk[u-1][0]+(2*x_max-1)*x_displace,walk[u-1][1]+(2*y_max-1)*y_displace
                    else:
                        x_last,y_last = origin
                    lift = (not wrap_line) and (abs(x-x_last) > 1 or abs(y-y_last) > 1) # the pointer wrapped around to the other side
                    if lift:
                        turtle.penup()
                    turtle.goto(x*dilate,y*dilate)
                    if lift:
                        turtle.pendown()
                turtle.penup()
        yf = origin[1]
    if border:
        bx = x_max-border_width/dilate
        by = y_max-border_width/dilate
        turtle.goto((x_max-border_width/dilate)*dilate,(yf)*dilate) # east side
        turtle.goto((bx*dilate,by*dilate)) # far northeast
        for j in range(border_thick):
            for x,y in [
                #(bx,by), # northeast
                (-bx,by), # northwest
                (-bx,-by), # southwest
                (bx,-by), # southeast
                (bx,by-1/dilate) # northeast to lead into next cycle by straight line rather than diagonal
                ]:
                turtle.goto(x*dilate-math.copysign(j,x),y*dilate-math.copysign(j,y))

    turtle.exitonclick()

def test():
    print(random_walk(3, repeat = False))

if __name__ == "__main__":
    #import numpy as np
    #import matplotlib.pyplot as plt
    import math, random
    
    main()
    #test()