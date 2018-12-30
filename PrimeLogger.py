import sympy

fp = "primes.txt"

try:
    with open(fp) as f:
        lines = f.readlines()
    last_prime = int(lines[-1].strip())
except FileNotFoundError:
    last_prime = 1

n_per_block = 10000  # buffer output instead of opening and closing file too much
while True:
    next_primes = []
    for _ in range(n_per_block):
        last_prime = sympy.nextprime(last_prime)
        next_primes.append(last_prime)
    with open(fp, "a") as f:
        for p in next_primes:
            f.write("{}\n".format(p))

