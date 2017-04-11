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
            func = self.turtle_dict[x]
            func()

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
        "A": LSystemRule("A", {"F": 0.5, "AB": 0.25, "A": 0.25}),
        "B": LSystemRule("B", {"ABA": 0.4, "BA": 0.3, "B": 0.3}),
    }
    koch_curve_rule_dict = {
        "F": LSystemRule("F", {"FLFRFLF": 1, "F": 0}),
    }
    turtle_dict = {
        "L": lambda: turtle.left(60),
        "R": lambda: turtle.right(120),
        "F": lambda: turtle.forward(1),
    }
    system = LSystem(koch_curve_rule_dict, turtle_dict)

    start_str = "F"
    res = system.apply_iterated(start_str, 20, max_length=10000)
    print(len(res))

    system.plot(res)