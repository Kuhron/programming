#FIXME the math is messed up for calculating terms

def get_somos(k, max_n=None):
    window = []
    i = 1  # initialize subscript from 1
    while max_n is None or i <= max_n:
        if i <= k:
            term = 1
            yield term
            window.append(term)
        else:
            assert len(window) == k
            # print("window:", window)
            denom = window[-k]
            numer = 0
            max_j = k//2
            for j in range(1, max_j+1):
                a_index = -k + j
                b_index = -j
                # print("j", j, a_index, b_index)
                a = window[a_index]
                b = window[b_index]
                numer += a * b
            term = numer / denom
            if term % 1 == 0:
                term = int(term)
            yield term
            window.append(term)
        window = window[-k:]
        i += 1


if __name__ == "__main__":
    k = 9
    max_n = 50
    for x in get_somos(k, max_n):
        print(x, end=", ")
    print()
