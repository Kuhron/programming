from datetime import datetime, timedelta
import random
import time


def time_expression_to_seconds(s):
    units = {"s":1, "min":60, "h":60*60, "d":60*60*24, "mo":60*60*24*365.24/12, "y":60*60*24*365.24}

    n, unit = s.split(" ")
    try:
        n = int(n)
    except ValueError:
        print("function time_expression_to_seconds: input must be [number][space][unit (1 letter)]")
        return None
    if unit not in units:
        print("function time_expression_to_seconds: input must be [number][space][unit (1 letter)]")
        return None

    return n * units[unit]


def get_destination(sgn, max_distance_seconds):
    return datetime.now() + sgn * timedelta(seconds=random.uniform(0, max_distance_seconds))


def main():
    now = time.time()
    direction = int(input("Would you like to go forward or backward in time, or either?\n\
        1. forward\n\
        2. backward\n\
        3. either\n"))
    if direction not in [1, 2, 3]:
        raise ValueError("invalid user input for direction of travel")
    max_distance = time_expression_to_seconds(input("What is the range of time you are considering?\n"
        "Example: \"4 min\" ([number] [s, min, h, d, mo, y]\n"
        "If you are going both directions, this amount is the maximum for either direction.\n"))

    if direction == 3:
        sgn = random.choice([-1, 1])
    else:
        sgn = 1 if direction == 1 else -1 if direction == 2 else None

    destination = get_destination(sgn, max_distance)
    dest_str = destination.strftime("%Y-%m-%d %H:%M:%S")
    print("You land at {0}. Have fun!".format(dest_str))


main()