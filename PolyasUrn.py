# inspired by https://www.technologyreview.com/s/603366/mathematical-model-reveals-the-patterns-of-how-innovations-arise/

import random

import matplotlib.pyplot as plt


class Ball:
    def __init__(self, color):
        self.color = color

    @staticmethod
    def get_random_color():
        n = 0
        power = 0
        while True:
            if random.random() < 0.5:
                n += 2**power
            if random.random() < 0.2:
                break
            power += 1
        return str(n)


class Urn:
    def __init__(self, contents=None):
        self.contents = contents if contents is not None else []
        self.colors = set(b.color for b in self.contents)
        self.seen_colors = {}

    def draw(self):
        ball = random.choice(self.contents)
        if ball.color in self.seen_colors:
            self.seen_colors[ball.color] += 1
            n = random.randint(1, 4)
            for i in range(n):
                self.add_ball(ball.color)
        else:
            self.seen_colors[ball.color] = 1
            n = random.randint(1, 4)
            for i in range(n):
                new_color = self.get_new_color()
                self.add_ball(new_color)

    def add_ball(self, color):
        self.contents.append(Ball(color))
        self.colors.add(color)

    def add_starting_balls(self, n):
        for i in range(n):
            self.add_ball(self.get_new_color())

    def get_new_color(self):
        while True:
            c = Ball.get_random_color()
            if c not in self.colors:
                return c

    def plot(self):
        keys = sorted(self.seen_colors, key=lambda k: -1*self.seen_colors[k])
        for i, key in enumerate(keys):
            count = self.seen_colors[key]
            plt.scatter(i+1, count, label=key)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("rank")
        plt.ylabel("occurrences")
        plt.show()


if __name__ == "__main__":
    urn = Urn()
    urn.add_starting_balls(5)
    for t in range(100000):
        urn.draw()
    urn.plot()
