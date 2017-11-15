import pandas as pd


def get_distance_matrix():
    borders = {}
    with open("StateAdjacency.txt") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        state, *neighbors = line.split(",")
        borders[state] = neighbors
    states = sorted(borders.keys())
    matrix = pd.DataFrame(index=states, columns=states)
    for state in states:
        matrix.ix[state, state] = 0
        for other in borders[state]:
            matrix.ix[state, other] = matrix.ix[other, state] = 1
    assert pd.isnull(matrix.ix["ME", "CA"])  # should not have been filled out yet
    for i in range(2, 100):  # should never get anywhere near 100
        if i > 90:
            raise RuntimeError("we have a problem! distance growing too large")
        print(i)
        number_was_used = False
        for state in states:
            most_recently_reached = [x for x in states if matrix.ix[state, x] == i - 1]
            for x in most_recently_reached:
                for y in borders[x]:
                    if pd.isnull(matrix.ix[state, y]):
                        matrix.ix[state, y] = matrix.ix[y, state] = i
                        number_was_used = True
        if not number_was_used:
            # every other cell is unreachable by any path in the graph, so make it a larger distance (i.e., the one that we just failed to use)
            for x in states:
                for y in states:
                    if pd.isnull(matrix.ix[x, y]):
                        matrix.ix[x, y] = matrix.ix[y, x] = i
            break

    return matrix


if __name__ == "__main__":
    m = get_distance_matrix()
    pd.DataFrame.to_csv(m, "StateDistanceMatrix.csv")