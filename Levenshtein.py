# just making a file for this since I use it so much, so I don't have to keep looking it up or rewriting it
# there is also the library python-Levenshtein (pip name)


# copied from https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
def levenshtein(seq1, seq2):
    big_length = 1000
    if max(len(seq1), len(seq2)) > big_length:
        raise Exception("Levenshtein function will be memory-intensive with long strings. Your strings have lengths {} and {}.".format(len(seq1), len(seq2)))
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    # matrix = np.zeros ((size_x, size_y))
    matrix = [[0 for y in range(size_y)] for x in range(size_x)]
    for x in range(size_x):
        matrix [x][0] = x
    for y in range(size_y):
        matrix [0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix [x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1][size_y - 1])


def normalized_levenshtein(x, y):
    d = levenshtein(x, y)
    max_d = max(len(x), len(y))
    return d / max_d


