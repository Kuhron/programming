import sympy


def factor_one_level(n):
    factors = sympy.ntheory.factorint(n)
    array = []
    for p, exp in sorted(factors.items()):
        prime_index = sympy.ntheory.generate.primepi(p) - 1
        if len(array) < prime_index + 1:
            array += [0] * (prime_index + 1 - len(array))
        array[prime_index] = exp
    return array[::-1]  # larger primes on the left


def integer_to_factor_notation(n):
    if n in [0, 1]:
        return str(n)
    array = factor_one_level(n)
    factored = factor_one_level(n)
    if sum(factored) == 1:
        # n is prime
        return "(" + "".join(str(x) for x in factored) + ")"
    else:
        return "(" + "".join(integer_to_factor_notation(x) for x in factored) + ")"


def factor_notation_to_integer(s):
    if s in ["0", "1"]:
        return int(s)





if __name__ == "__main__":
    # n = (2**6)*(3**4)*(5**5)
    # n = 16 * 3
    n = 
    print(integer_to_factor_notation(n))
    # s = "((1)0)"
    s = "1(1)"
    print(factor_notation_to_integer(s))