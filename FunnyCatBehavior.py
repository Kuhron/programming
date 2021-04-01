import random

verbs = ["lick", "jump onto", "scratch", "eat", "knock off", "hiss at"]
nouns = ["plastic", "metal", "counter", "wood", "paper", "toys", "decorations", "TV"]


def get():
    return random.choice(verbs) + " " + random.choice(nouns)


if __name__ == "__main__":
    print(get())
