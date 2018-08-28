import random
import turtle


class LSystem:
    def __init__(self, rule_dict, turtle_dict):
        self.rule_dict = rule_dict
        assert all(type(x) is LSystemRule for x in self.rule_dict.values())
        self.turtle_dict = turtle_dict

    def apply(self, s):
        result = ""
        for c in s:
            rule = self.rule_dict.get(c)
            if rule is None:
                result += c
            else:
                result += rule.apply(c)
        return result

    def apply_iterated(self, start_str, n_iterations, max_length=10000):
        result = start_str
        for i in range(n_iterations):
            result = self.apply(result)[:max_length]
        return result

    def plot(self, s):
        turtle.hideturtle()
        turtle.speed(0)

        for x in s:
            instructions = self.turtle_dict[x]
            for instruction in instructions:
                if instruction[0] == "-":
                    negative = True
                    instruction = instruction[1:]
                code = instruction[0]
                num = int(instruction[1:])
                if code == "L":
                    turtle.left(num)
                elif code == "R":
                    turtle.right(num)
                elif code == "F":
                    turtle.forward(num)
                else:
                    raise ValueError("invalid instruction code {}".format(code))

        fp = "LSystemImage.ps"
        turtle.getscreen().getcanvas().postscript(file=fp)
        print("saved image to " + fp)

        turtle.exitonclick()


class LSystemRule:
    # no environment conditioning for now

    def __init__(self, source, dest_prob_dict):
        self.source = source
        self.dest_prob_dict = dest_prob_dict
        assert self.source in self.dest_prob_dict
        self.total_prob = sum(self.dest_prob_dict.values())

    def apply(self, s):
        assert s == self.source
        dest = self.select_dest()
        return dest

    def select_dest(self):
        index = random.random() / self.total_prob
        cands = (x for x in self.dest_prob_dict.items())  # should be generator but just in case (Python 2)
        while True:
            cand, prob = next(cands)
            if index < prob:
                return cand
            index -= prob

        raise Exception("failed to choose destination string")



if __name__ == "__main__":
    rule_dict = {
        "A": LSystemRule("A", {">": 0.5, "AB": 0.25, "A": 0.25}),
        "B": LSystemRule("B", {"ABA": 0.4, "BA": 0.3, "B": 0.3}),
    }
    koch_curve_rule_dict = {
        # ">": LSystemRule(">", {">d>H>d>": 1, ">": 0}),
    }
    acoma_fractal_rule_dict = {
        ">": LSystemRule(">", {"c>F>>f>C": 1, ">": 0}),  # turns - into /\\/
    }
    turtle_dict = {
        "a": ["L15"],
        "b": ["L30"],
        "c": ["L45"],
        "d": ["L60"],
        "e": ["L75"],
        "f": ["L90"],
        "g": ["L105"],
        "h": ["L120"],
        "i": ["L135"],
        "j": ["L150"],
        "k": ["L165"],
        "l": ["L180"],
        "A": ["R15"],
        "B": ["R30"],
        "C": ["R45"],
        "D": ["R60"],
        "E": ["R75"],
        "F": ["R90"],
        "G": ["R105"],
        "H": ["R120"],
        "I": ["R135"],
        "J": ["R150"],
        "K": ["R165"],
        "L": ["R180"],
        ">": ["F1"],
        # "L": ["L60", ">1"],
        # "R": ["R120", ">3"],
        # ">": ["R60", ">1", "L60"],
    }
    system = LSystem(acoma_fractal_rule_dict, turtle_dict)

    start_str = ">"
    res = system.apply_iterated(start_str, 6, max_length=10000)
    print(len(res))

    system.plot(res)
