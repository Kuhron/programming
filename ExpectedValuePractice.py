# protip: if you have two values with similar weights
# to get expected value from average, use this general eqn:
# given (m: w_m), (n: w_n)
# e.g. (5: 1.2), (13: 1.5)
# expected value E = (w_m m + w_n n)/(w_m + w_n)
# average A = (m + n)/2
# difference d = E - A = [algebra] (w_m - w_n)/(w_m + w_n) * (m - n)/2
# so find the weights' diff over sum, multiply by half the numbers' diff
# here, d = (-0.3)/(2.7) * (-8)/2 = 1/9 * 4 = 0.4444...
# add this to the average (9) to get E = 9.4444...

def expected_value(dct):
    """
    Takes a dictionary of numerical outcomes and their relative probabilities,
    returns the expected value. Probabilities need not sum to 1.
    """
    result = 0.0
    total_prob = float(sum([dct[a] for a in dct]))
    for x in dct:
        result += x*dct[x]/total_prob
    return result

def gen_dct():
    result = {}
    for i in range(random.randrange(2,6)):
        h = min(max(math.floor(random.normalvariate(10,5)),0),20)
        while h in result:
            h = min(max(math.floor(random.normalvariate(10,5)),0),20)
        result[h] = random.randrange(1,21)/10.0
    return result

def run_guess():
    d = gen_dct()
    for x in d:
        if x < 10:
            print("", x, ":", d[x])
        else:
            print(x, ":", d[x])
    ans = expected_value(d) # returns float
    guess = float(input("What is the expected value? "))
    print("{0:.2f}% off. Answer: {1:.2f}\n".format(100*(guess-ans)/ans, ans))
    
if __name__ == "__main__":
    import math, random

    while True:
        run_guess()
