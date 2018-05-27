import argparse
import time

import matplotlib.pyplot as plt

from CrystalGrowthRules import CrystalGrowthRules as CGR


class States:
    EXISTING = 2
    NEW = 1
    EMPTY = 0

    # STR = " OX"
    STR = " ▪▫"

    @staticmethod
    def get_char(state):
        return States.STR[state]


class Grid:
    def __init__(self, side_length):
        self.side_length = side_length
        self.grid = [[States.EMPTY for i in range(side_length)] for j in range(side_length)]
        self.birth_grid = [[float("nan") for i in range(side_length)] for j in range(side_length)]
        self.points_by_state = {state: set() for state in range(100)}  # please don't use more than 100 states
        self.points_by_state[States.EMPTY] = {(i, j) for j in range(side_length) for i in range(side_length)}
        self.state_ordering = [States.EMPTY, States.NEW, States.EXISTING]
        self.iteration = 0

    def get_state_at(self, point):
        return self.grid[point[0]][point[1]]

    def set_state_at(self, point, new_state, maintain_ordering=True):
        old_state = self.get_state_at(point)
        self.points_by_state[old_state].remove(point)
        new_state = self.get_higher_state(old_state, new_state)
        self.grid[point[0]][point[1]] = new_state
        if old_state == 0 and new_state > 0:
            self.birth_grid[point[0]][point[1]] = self.iteration
        self.points_by_state[new_state].add(point)

    def set_state_at_point_array(self, point_array, new_state_array):
        for p_row, s_row in zip(point_array, new_state_array):
            for point, state in zip(p_row, s_row):
                self.set_state_at(point, state)

    def get_higher_state(self, s1, s2):
        return max([s1, s2], key=lambda x: self.state_ordering.index(x))

    def grow(self, growth_rules):
        points_to_grow = [p for p in self.points_by_state[States.NEW]]  # avoid "set changed size during iteration"
        if points_to_grow == []:
            raise StopGrowthIteration

        for point in points_to_grow:
            # do this before the new points are added, since the ones growing are considered "existing" for the purposes of determining environments
            self.set_state_at(point, States.EXISTING)

        for point in points_to_grow:
            neighbors = self.get_neighbors(point)
            neighbor_states = self.get_state_array(neighbors)
            # grow the neighbors depending on the rules

            # NEW WAY
            env_str = GrowthRule.convert_environment_to_rule_input_code(neighbor_states)
            # print("looking up rule for str {}".format(env_str))
            rule = growth_rules.rule_by_env_str.get(env_str)
            # print("got rule {}".format(rule))
            if rule is not None:
                self.set_state_at_point_array(neighbors, rule.resulting_environment)

            
            # OLD WAY, slow because it iterates over each rule for each point. instead just go get the correct rule, if any, directly by putting a dict of environment code str to rule object in the GrowthRuleSet class
            # rule_matches = [(env, rule) for env, rule in growth_rules.rules.items() if rule.applies(neighbor_states)]
            # 
            # if len(rule_matches) == 1:
            #     # apply growth rule
            #     k, rule = rule_matches[0]
            #     # print("rule applied at point {} with environment\n{}\nand resulting environmment\n{}".format(point, neighbor_states, resulting_environment))
            #     self.set_state_at_point_array(neighbors, rule.resulting_environment)
            # elif len(rule_matches) > 1:
            #     raise Exception("should not match more than one rule because GrowthRuleSet's rules should have at most one rule for each environment\nrule matches: {}".format(rule_matches))
            # else:
            #     # do nothing, just mark the current point as existing
            #     # print("no rule applied at point {} with environment\n{}".format(point, neighbor_states))
            #     pass

        self.iteration += 1

    def get_neighbors(self, point):
        # still includes point's coordinates too; this should not be changed, so we are dealing with 3x3 arrays as much as possible
        x, y = point
        n = self.side_length
        return [
            [((x-1)%n, (y-1)%n), ((x-1)%n, (y  )%n), ((x-1)%n, (y+1)%n)],
            [((x  )%n, (y-1)%n), ((x  )%n, (y  )%n), ((x  )%n, (y+1)%n)],
            [((x+1)%n, (y-1)%n), ((x+1)%n, (y  )%n), ((x+1)%n, (y+1)%n)],
        ]

    def get_state_array(self, point_array):
        return [[self.get_state_at(p) for p in row] for row in point_array]

    def print(self):
        if self.side_length > 37:
            # too big to fit on screen
            print("can't fit grid on screen")
            return
        print("/" + "-" * (2 * self.side_length - 1) + "\\")
        for row in self.grid:
            print("|" + " ".join(States.get_char(state) for state in row) + "|")
        print("\\" + "-" * (2 * self.side_length - 1) + "/")

    def plot_age(self):
        plt.imshow(self.birth_grid)
        plt.colorbar()
        plt.show()


class StopGrowthIteration(Exception):
    pass


class GrowthRuleSet:
    def __init__(self):
        self.rules = {}
        self.rule_by_env_str = {}

    def add(self, rule):
        key = rule.existing_environment
        value = rule.resulting_environment
        equivalent_keys = get_rotations_and_reflections(key)
        equivalent_values = get_rotations_and_reflections(value)
        # ordering of keys and values should correspond, given that the function generating them maintains ordered output
        for k, v in zip(equivalent_keys, equivalent_values):
            equivalent_rule = GrowthRule(k, v)
            self.rules[array_to_tuple(k)] = equivalent_rule
            self.rule_by_env_str[GrowthRule.get_code_str_from_environment(k)] = equivalent_rule
        # make sure original rule overwrites any that have replaced it due to having identical key, but possibly different value
        self.rules[array_to_tuple(key)] = rule
        self.rule_by_env_str[GrowthRule.get_code_str_from_environment(key)] = rule
        
        # debug
        # print("\ndict:")
        # for k, v in sorted(self.rule_by_env_str.items()):
        #     print("env str {} : rule {}".format(k, v.get_code_str()))
        # print("\n")

    def print_codes(self, output_path=None):
        to_print = self.rules.values()
        code_strs = [rule.get_code_str() for rule in to_print]
        code_strs = sorted(code_strs)
        assert len(code_strs) == len(self.rules)
        s = "\n".join(code_strs)
        if output_path is None:
            print(s)
        else:
            with open(output_path, "w") as f:
                f.write(s)


class GrowthRule:
    def __init__(self, existing_environment_1s_and_0s, resulting_environment_1s_and_0s):
        # print("initializing rule with input\n{} and\n{}".format(existing_environment_1s_and_0s, resulting_environment_1s_and_0s))
        in_1_0 = GrowthRule.remove_central_value(existing_environment_1s_and_0s)
        out_1_0 = GrowthRule.remove_central_value(resulting_environment_1s_and_0s)
        self.existing_environment = GrowthRule.convert_1s_and_0s_to_state(in_1_0, States.EXISTING)
        self.resulting_environment = GrowthRule.convert_1s_and_0s_to_state(out_1_0, States.NEW)
        self.existing_environment_code_str = GrowthRule.get_code_str_from_environment(self.existing_environment)
        # print("new vars:\n\nin 1 0\n{}\n\nout 1 0\n{}\n\nex env\n{}\n\nres env\n{}\n\n".format(in_1_0, out_1_0, self.existing_environment, self.resulting_environment))
        # print("code: {}\n\n".format(self.get_code_str()))

    def applies(self, environment):
        # trying to deprecate this function in favor of getting the environment code for each spot first, then just getting the rule that applies to it by looking up the environment code str in a dict within GrowthRuleSet, to avoid iterating over rules for every point to see which one applies
        environment_code = GrowthRule.convert_environment_to_rule_input_code(environment)
        return self.applies_to_environment_code_str(environment_code)

    def applies_to_environment_code_str(self, environment_code):
        return environment_code == self.existing_environment_code_str

    @staticmethod
    def convert_environment_to_rule_input_code(environment):
        environment = GrowthRule.remove_central_value(environment)
        environment = GrowthRule.filter_state_array(environment, States.EXISTING)
        environment_code = GrowthRule.get_code_str_from_environment(environment)
        return environment_code
        

    @staticmethod
    def convert_1s_and_0s_to_state(input_arr, output_state):
        # changed condition from == 1 to != 0 because the rules have already been initialized (and thus their 1s converted to >= 1) before GrowthRuleSet.add is called, and don't want the existing environment to be overwritten to all zeros because the 1s were already changed to 2s
        return [[output_state if cell != 0 else 0 for cell in row] for row in input_arr]

    @staticmethod
    def filter_state_array(input_arr, state_to_keep):
        return [[x if x == state_to_keep else 0 for x in row] for row in input_arr]

    @staticmethod
    def remove_central_value(input_arr):
        # replaces value in center of array with 0 so state at point does not interfere with neighbor environment classification
        assert len(input_arr) == 3 and all(len(x) == 3 for x in input_arr)
        return [[input_arr[i][j] if (i, j) != (1, 1) else 0 for j in range(3)] for i in range(3)]

    # guidelines for rules:
    # - middle cell should be States.NEW (the one that is being grown, because it hasn't yet done so due to just having sprouted from an older growth)
    # - pay attention only to States.EXISTING for the environment; do not pay attention to new cells, so that each iteration of growth occurs simultaneously

    @staticmethod
    def get_code_str_from_environment(environment):
        return "".join("".join(str(x) for x in row) for row in environment)

    def get_code_str(self):
        return GrowthRule.get_code_str_from_environment(self.existing_environment) + " -> " + GrowthRule.get_code_str_from_environment(self.resulting_environment)

    @staticmethod
    def get_rule_from_code(s):
        # print("creating rule from code {}".format(s))
        inp, outp = s.split(" -> ")
        assert len(inp) == len(outp) == 3 ** 2
        str_to_arr = lambda s: [[int(c) for c in s[3 * n : 3 * (n + 1)]] for n in range(3)]
        inp_arr = str_to_arr(inp)
        outp_arr = str_to_arr(outp)
        return GrowthRule(inp_arr, outp_arr)


def get_rotations_and_reflections(arr):
    # output must be ordered
    result = []
    for rotation in get_rotations(arr):
        result += [get_reflection(rotation), rotation]
    return result


def get_rotations(arr):
    f = rotate_right
    # keep original array last in the whole thing so it overwrites the others and gives the intended output if the input is symmetrical
    return [f(arr), f(f(arr)), f(f(f(arr))), arr]  # don't optimize prematurely


def rotate_right(arr):
    assert len(arr) == 3 and all(len(x) == 3 for x in arr)  # whatever, generalize it later if you need to
    return [
        [arr[2][0], arr[1][0], arr[0][0],],
        [arr[2][1], arr[1][1], arr[0][1],],
        [arr[2][2], arr[1][2], arr[0][2],],
    ]


def get_reflection(arr):
    # just use one axis; if you want reflection along the other, use rotations as well
    return [row[::-1] for row in arr]


def is_equivalent(arr1, arr2):
    return any(arr1 == x for x in get_rotations_and_reflections(arr2))


def array_to_tuple(arr):
    return tuple(tuple(x for x in row) for row in arr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expedite", dest="expedite", action="store_true")
    parser.add_argument("-n", "--side_length", dest="side_length", type=int, default=37)
    parser.add_argument("-r", "--rule_set", dest="rule_set", type=str)
    args = parser.parse_args()

    n = args.side_length
    c = int((n-(n%2))/2)  # center
    grid = Grid(n)

    seed_position_array_1 = [
        [(c, c),],
    ]
    seed_position_array_2 = [
        [(c-1, c),],
        [(c, c-1), (c, c+1),],
    ]

    seed_position_array = seed_position_array_1
    seed_state_array = [[States.NEW for cell in row] for row in seed_position_array]

    grid.set_state_at_point_array(seed_position_array, seed_state_array)

    growth_rules = GrowthRuleSet()

    if args.rule_set:
        with open(args.rule_set) as f:
            lines = f.readlines()
        rule_set = [GrowthRule.get_rule_from_code(line.strip()) for line in lines]
    else:
        # rule_set = CGR.original_rules
        # rule_set = CGR.diamond_rules
        # rule_set = CGR.test_directionality_rules
        rule_set = CGR.generate_random_rules()

        rule_set = [GrowthRule(*rule) for rule in rule_set]  # expect pair of arrays, for input and output
        rule_set = sorted(rule_set, key=lambda x: x.get_code_str())

    for rule in rule_set:
        # print(rule)
        if type(rule) is not GrowthRule:
            rule = GrowthRule(*rule)
        growth_rules.add(rule)

    grid.print()
    # for i in range(10):
    while True:
        try:
            # input("\npress enter to continue\n")
            grid.grow(growth_rules)
            if not args.expedite:
                grid.print()
                time.sleep(0.1)
        except StopGrowthIteration:
            grid.print()
            print("No more points to grow!")
            break

    if not args.expedite and grid.iteration > 5:
        input("press enter to continue")

    if grid.iteration > 5:
        # print("rules generating this pattern:")
        # growth_rules.print_codes()
        growth_rules.print_codes("CrystalRules/last.txt")
        grid.plot_age()
