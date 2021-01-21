is_mult = lambda x: x % 3 == 0 or x % 5 == 0
mults = lambda n: filter(is_mult, range(1, n))  # mults(10) doesn't include 10 itself
sum_mults = lambda n: sum(mults(n))

assert sum_mults(10) == 23
print(sum_mults(1000))
