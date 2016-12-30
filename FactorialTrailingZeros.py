#apparently this already exists in a module
"""
def factorial(x):
    if x < 0 or x % 1 != 0:
        return "Nope"
    elif x == 0:
        return 1
    else:
        k = 1
        for u in range(1,x+1):
            k = k*u
        return k
"""
from math import factorial

#NUMERICAL METHOD FOR FTZ
"""
def trailing_zeros(x):
    i = 0
    while (x/(10.0**(i+1))) % 1.0 == 0.0 and (x/(10.0**i)) > 1.0:
        i += 1
    return i
"""

#STRING METHOD FOR FTZ
def trailing_zeros(x):
    s = str(x)
    i = 1
    while s[len(s)-i] == "0":
        i += 1
    return i-1

def ftz(x):
    return trailing_zeros(factorial(x))

"""
#overtakes linear growth between x=21 and x=22 (this difference maxes at f(x)=17)
for i in range(1,101):
    print(i-ftz(i))
"""

"""
#never overtakes quadratic
for i in range(1,101):
    print((i**2)-ftz(i))
"""

"""
#n log n, appears to increase steadily, at a rate which is either constant in the long run or grows rather slowly (step size ~ 3), overflow around x=150
from math import log
for i in range(101,201):
    print((i * log(i))-ftz(i))
"""

"""
#n log n - k*n, just a wild extrapolation of the previous finding, let's see what happens
from math import log
from math import pi
k = 3.4 #change this to experiment
for i in range(1,500):
    print((i * log(i))-(k*i)-ftz(i))
#pi gives a close following for a while, but ftz ultimately cannot overtake it, thus the ideal k is greater than pi
#with similar trial and error, we see that k* lies in the range (3.3,3.6)
#although notice with k=3.6, we see a global minimum being reached, so it seems that this proposed growth rate is still too fast for ftz
#so this "k*" really has no meaning / doesn't exist, since I was judging whether k was too high or low based on the behavior of the difference toward
    #the end of the range, right before the overflow
"""

# find a growth rate faster than linear, but slower than n log n - k*n
# go to bed you have work in the morning yo

from math import log
from math import e

def f(x):
    return x/4.0 - 2
    # ding ding ding we have a winner
    # even though the constant is uncertain,
    # the denominator is so 4
    # note that anomalous deviations seem to occur pretty much always,
    # but it looks like the trend is the same even for values of
    # in the hundred thousands

diffs = {}
for i in range(1,1000):
    diff = (f(i) - ftz(i))
    if diff not in diffs:
        diffs[diff] = 1
    else:
        diffs[diff] += 1
for k in range(-10,11):
    if k in diffs:
        print(k/4.0, diffs[k/4.0])
    else:
        print(k/4.0, 0)
print("----")
for k in sorted(diffs):
    print(k, diffs[k])
#why isn't it recognizing that those keys are in the dict
