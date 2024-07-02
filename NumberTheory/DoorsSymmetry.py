# looking at the symmetrical pattern about the log(x)/2 line in Doors.py

import numpy as np
import sympy

divs = divs = lambda x: sympy.divisors(x)
divs2 = lambda x: (divs(x)) if x**0.5 % 1 == 0 else (sorted([x**0.5] + divs(x)))

def print_chart(n):
    print(f"{n = }")
    for x in divs2(n):
        print((f"{x:>6}" if x % 1 == 0 else f"~{x:>5.1f}") + f" | {np.log(x) - np.log(n)/2:>4.2f}")
    print()


if __name__ == "__main__":
    ns = range(1685, 1723)
    # there is a trumpet-bell shape in the graph in this range (divisors getting closer and closer to sqrt(n) in a regular way)
    for n in ns:
        print_chart(n)
