def f1(x):
    return x**(1.0/x)
    
def ff2(x,n):
    a = x
    for i in range(n):
        a = f1(a)
    return a

def f2(x):
    return ff2(x,x)
    
def f3(x):
    return 1.0/(f2(x)-1)

def f4(x):
    return f3(x)/float(x)

#print(f4(94500000))

def sgn(x):
    if x == 0:
        return 0
    return int(abs(x)/float(x))

def find_target(f, target, start, step): # just for this special case
    initial_direction_set = False
    while True:
        if not initial_direction_set:
            direction = "undefined as of yet"
        print("Using start = {0}, step = {1}, direction = {2}".format(start,step,direction))
        val = f(start)
        print("Function evaluated at {0} gives {1:.7f}, which is {2:.7f} away from the target.".format(start, val, val-target))

        if abs(val-target) < 10**-7 or step < 1:
            print("Target found! x = {0}, f(x) = {1}".format(start,val))
            return

        if not initial_direction_set:
            if val>target:
                direction = 1
            else:
                direction = -1
            initial_direction_set = True    
    
        if sgn(val-target) == -1*direction: # this function has negative derivative
            # so we must switch the direction of search once the diff is the same as step direction
            direction = -1*direction
            step = step/2.0
            print("new step size: {0}".format(step))

        start = int(start + direction*step)

find_target(f4, 1, 94906266, 200000)


















