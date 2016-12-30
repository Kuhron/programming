# genetic algorithm development workspace
# start off with the task of maximizing entropy in the string
# want most equal accuracy of various predicting functions for next bit

# basic functionality

max_length = 200

def average(lst):
    # WHY IS THIS NOT IN math?
    return sum(lst)/float(len(lst))

def var(lst):
    m = average(lst)
    ss = 0.0
    for i in lst:
        ss += (i-m)**2
    return ss

# genetic functionality

def rate(b):
    # smaller variance of predictor accuracies gets better score
    # note that if the predictors are too similar, then we are not
    # actually maximizing entropy, as they may all be right or wrong

    accs = [p.accuracy(b) for p in predictors]
    #return 1.0/math.log(1.0+statistics.variance(accs))
    return 1.0/statistics.variance(accs)

def insertion(b):
    index = random.randrange(len(b))
    return b[:index] + str(random.randrange(2)) + b[index:]

def deletion(b):
    index = random.randrange(len(b))
    return b[:index] + b[index+1:]

def mutation(b):
    index = random.randrange(len(b))
    if b[index] == "0":
        a = "1"
    elif b[index] == "1":
        a = "0"
    return b[:index] + a + b[index+1:]

def bit():
    return str(random.randrange(2))

def asexually_reproduce(b):

    # this version allows each bit to undergo mutation of various kinds
    result = ""
    for i in range(len(b)):
        q = random.randrange(24)
        if q < 2: # insertion before
            result += bit() + b[i]
        elif q < 4: # insertion after
            result += b[i] + bit()
        elif q < 8: # mutation
            if b[i] == "0":
                result += "1"
            elif b[i] == "1":
                result += "0"
            else:
                print("You have a non-binary string!")
        elif q < 12: # deletion
            pass
        else: # keep it the same (1/2 probability)
            result += b[i]
        # truncate long strings
        if len(result) >= max_length:
            result = result [:max_length]
            break
    return result

    # this version gave at most one mutation per step
    """
    q = random.randrange(10)
    if q < 6:
        return insertion(b)
    elif q < 9:
        return mutation(b)
    else:
        return deletion(b)
    """

# predictors

class Predictor:
    # use gambler's fallacy if context > 0 (can never be < 0)
    def __init__(self, context, response_to_small, response_to_large):
        self.context = context
        self.response_to_small = response_to_small
        self.response_to_large = response_to_large

    def observe(self, b):
        if len(b) < self.context:
            # shouldn't be called in this instance
            print("A predictor is being used on a string that is too short.")
            return -1
        return average([int(b[i]) for i in range(len(b)-self.context, len(b))])
    
    def predict(self, b):
        if self.response_to_small == self.response_to_large:
            return self.response_to_small
        observation = self.observe(b)
        if observation == 0.5 or observation < 0 or observation > 1:
            # in the case of 0.5,
            # don't want to make a decision
            # always get it wrong
            return -1
        elif observation < 0.5:
            return self.response_to_small
        elif observation > 0.5:
            return self.response_to_large

    def accuracy(self, b):
        score = 0.0
        j = 0
        for i in range(self.context, len(b)):
            j += 1
            prediction = self.predict(b[:i])
            #print("predicted {0}, got {1}".format(prediction, b[i]))
            if prediction == int(b[i]):
                score += 1.0
        if j == 0:
            return -1
        return score/j

p_0 = Predictor(0, 0, 0)
p_1 = Predictor(0, 1, 1)
p_same = Predictor(1, 0, 1)
p_opposite = Predictor(1, 1, 0)
# avoid predictors where 0.5 can be observed
# too lazy to decide what to do about it
p_same_3 = Predictor(3, 0, 1)
p_opposite_3 = Predictor(3, 1, 0)

predictors = [p_0, p_1, p_same, p_opposite, p_same_3, p_opposite_3]

# simulation functions

def simulate_asexual_reproduction():
    already_seen = {}
    for i in range(max_length+1):
        already_seen[i] = []
    b = "".join([str(random.randrange(2)) for u in range(50)])
    already_seen[50].append(b)
    print("Init String: {0} (rated {1:.2f})".format(b, rate(b)))
    for i in range(10000000):
        c = asexually_reproduce(b)
        if c not in already_seen[len(c)]:
            already_seen[len(c)].append(c)
            if rate(c) > rate(b):
                b = c
                print("Next String: {0} (rated {1:.2f} at step {2})".format(b, rate(b), i))

# other / misc

def theoretical_best_rating(length):
    best = 0.0
    for i in range(2**(length-1), 2**length):
        rating = rate("{0:b}".format(i))
        if rating > best:
            best = rating
    return best

def show_theoretical_best_ratings():
    ratings = []
    for u in range(1,16):
        t = theoretical_best_rating(u)
        ratings.append(t)
        print("length", u, "gives", t)

####### MAIN METHOD #######

if __name__ == "__main__":
    import math
    import random
    import statistics

    simulate_asexual_reproduction()
    #show_theoretical_best_ratings()
