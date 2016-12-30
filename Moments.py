import random

import matplotlib.pyplot as plt


def m(lst, n):
    """
    nth sample moment of a list
    """
    if n <= 0 or type(n) is not int:
        return None
    elif n == 1:
        return sum(lst) *1.0/ len(lst)
    elif n == 2:
        mm = m(lst, 1)
        return m([(i-mm)**2 for i in lst], 1)
    else:
        standardize = n > 2
        mm = m(lst, 1)
        s = m(lst, 2)
        q = 1.0/(s**n) if standardize else 1
        return q * m([(i-mm)**n for i in lst], 1)

def r(n):
    return [dist() for i in range(n)]

def flip(lst):
    """
    returns list [k_i] of the first odd monents (excluding the mean) after which the sign flips
    """
    k_max = 1000
    a = []
    # mm = m(lst, 1)
    # s = 1 if mm > 0 else -1 if mm < 0 else 0
    s = None
    try:
        for k in range(1, k_max+1, 2):
            q = m(lst, k)
            t = 1 if q > 0 else -1 if q < 0 else 0
            if t != s:
                a.append(k)
                s = t
    except OverflowError:
        # sk = str(k)[-1]
        # ordinal = "st" if sk == "1" else "nd" if sk == "2" else "rd" if sk == "3" else "th"
        # print("Overflow reached at {0}{1} moment. Returning flips found up to there.".format(k, ordinal))
        pass
    return a

def estimate_expected_mean_of_flips(n_lsts, n_per_lst):
    sm = 0
    for i in range(n_lsts):
        lst = r(n_per_lst)
        fl = flip(lst)
        sm += m(fl, 1)
    # calculating mean "manually" here rather than storing a bunch of lists
    return sm *1.0/ n_lsts

def dist():
    return random.normalvariate(0, 1)
    # return random.paretovariate(10) * random.choice([-1, 1])


lst = r(10**4)
try:
    for k in range(1, 8):
        pass # print(m(lst, k))
except OverflowError:
    print("Overflow reached")
fl = flip(lst)
print(fl)
print(m(fl, 1))
plt.hist(lst, bins=100)
plt.show()

# ms = [m(flip(r(10000)),1) for i in range(100)]
# print(m(ms, 1))
# plt.hist(ms)
# plt.show()
print(estimate_expected_mean_of_flips(n_lsts=10, n_per_lst=10**4))

# a = []
# for i in range(100):
#     a.extend(flip(r(10000)))
# plt.hist(a)
# # plt.yscale("log", nonposy="clip") # fails to show bars when something has 1 occurrence; tried http://stackoverflow.com/questions/17952279/
# plt.show()

# n_lsts = 100
# n_per_lst = 10000
# print("estimates for moments of dist() (on one draw of {0} lists of {1} observations each)".format(n_lsts, n_per_lst))
# ls = [r(n_per_lst) for i in range(n_lsts)]
# for k in range(1, 8):
#     mm = m([m(lst, k) for lst in ls], 1)
#     print("{0}: {1}".format(k, mm))















