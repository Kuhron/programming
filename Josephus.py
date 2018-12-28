import matplotlib.pyplot as plt


def f(n):
    designations = list(range(1, n+1))
    statuses = [1 for person in designations]
    current_index = 0
    increment = lambda index: (index + 1) % n
    while sum(statuses) > 1:
        assert statuses[current_index] == 1  # current person must be alive
        next_index = increment(current_index)
        while statuses[next_index] == 0:
            next_index = increment(next_index)
        # now kill this person
        statuses[next_index] = 0
        # now get next living person as current index, note that they will not be any of the ones we just checked, nor the one that just died
        # if current index is the last one then it will loop around to them again; can only infinite loop if no one is alive
        current_index = increment(next_index)
        while statuses[current_index] == 0:
            current_index = increment(current_index)
        # go again, this current index kills next living person, unless they are the last one
    return designations[current_index]


# later may be able to use memoized results for smaller n
# for instance, say in the case of N you have whittled it down to four people left with designations [11, 12, 205, 894]
# then you can just look up the answer for n=4 and get that member of this list (depending on whose turn it is)
# so you can actually just use f(n-1) and build up that way


if __name__ == "__main__":
    for i in range(1, 20):
        print(i, f(i))
