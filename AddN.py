import random

digs = "0123456789"

def get_dig():
    return random.choice(digs)

def get_num(length):
    return "".join(get_dig() for _ in range(length))

def add_n(char, n):
    return str((int(char) + n) % 10)

def add_n_to_num(num, n):
    return "".join(add_n(char, n) for char in num)


if __name__ == "__main__":
    length = 4
    n = 3
    successes = 0
    trials = 0

    while True:
        num = get_num(length)
        print(num)
        print("add {}!".format(n))
        inp = input()
        ans = add_n_to_num(num, n)
        if inp == ans:
            successes += 1
            print("correct!")
        else:
            print("doh! the correct answer was {}".format(ans))
        trials += 1
        print("score: {}/{} ({}%)\n".format(successes, trials, int(100 * successes / trials)))
