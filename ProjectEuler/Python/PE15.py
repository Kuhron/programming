import math
f = math.factorial
c = lambda n, k: f(n)/(f(n-k)*f(k))

n = 20
print(c(2*n, n))  # string will be all r's and d's (right and down) of length 2n, and exactly n of those can be r's
