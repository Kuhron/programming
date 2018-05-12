class States:
    EXISTING = 2
    NEW = 1
    EMPTY = 0


class Grid:
    def __init__(self, side_length):
        self.side_length = side_length
        self.grid = [[States.EMPTY for i in range(side_length)] for j in range(side_length)]

    def get_state_at(self, point):
        return self.grid[point[0]][point[1]]

    def set_state_at(self, point, new_state):
        self.grid[point[0]][point[1]] = new_state

    def grow(self, growth_rules):
        for point in self.points_by_state[States.NEW]:
            neighbors = self.get_neighbors(point)
            neighbor_states = self.get_state_array(neighbors)
            # grow the neighbors depending on the rules
            if any(is_equivalent(neighbor_states, x) for x in growth_rules.keys()):
                ?
            else:
                # do nothing, just mark the current point as existing
            self.set_state_at(point, States.EXISTING)

    def get_neighbors(self, point):
        x, y = point
        n = self.side_length
        return [
            [((x-1)%n, (y-1)%n), ((x  )%n, (y-1)%n), ((x+1)%n, (y-1)%n)],
            [((x-1)%n, (y  )%n), ((x  )%n, (y  )%n), ((x+1)%n, (y  )%n)],
            [((x-1)%n, (y+1)%n), ((x  )%n, (y+1)%n), ((x+1)%n, (y+1)%n)],
        ]

    def get_state_array(self, point_array):
        return [[self.get_state_at(p) for p in row] for row in point_array]


class GrowthRules:
    pass


def get_rotations_and_reflections(arr):
    result = []
    for rotation in get_rotations(arr):
        result += [rotation, get_reflection(rotation)]
    return result


def get_rotations(arr):
    f = rotate_right
    result = [arr, f(arr), f(f(arr)), f(f(f(arr)))]  # don't optimize prematurely


def rotate_right(arr):
    assert len(arr) == 3 and all(len(x) == 3 for x in arr)  # whatever, generalize it later if you need to
    return [
        arr[2][0], arr[1][0], arr[0][0],
        arr[2][1], arr[1][1], arr[0][1],
        arr[2][2], arr[1][2], arr[0][2],
    ]


def get_reflection(arr):
    # just use one axis; if you want reflection along the other, use rotations as well
    return [row[::-1] for row in arr]


def is_equivalent(arr1, arr2):
    return any(arr1 == x for x in get_rotations_and_reflections(arr2))


def get_equivalence



if __name__ == "__main__":
    pass
