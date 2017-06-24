import sympy

import LSystem


# parentheses surrounding a group means it is in prime factor notation
# ultimately all ints except 0 can be written in factor notation


def factor_one_level(n):
    if n == 1:
        return [0]  # otherwise sympy will just return an empty array; I'll use a convention where 1 is "(0)" rather than "()"
    factors = sympy.ntheory.factorint(n)
    array = []
    for p, exp in sorted(factors.items()):
        prime_index = sympy.ntheory.generate.primepi(p) - 1
        if len(array) < prime_index + 1:
            array += [0] * (prime_index + 1 - len(array))
        array[prime_index] = exp
    return array[::-1]  # larger primes on the left


def integer_to_factor_notation(n, factor_one=True):
    if n == 0:
        return "0"
    if n == 1 and not factor_one:
        return "1"
    array = factor_one_level(n)
    factored = factor_one_level(n)
    return "(" + "".join(integer_to_factor_notation(x, factor_one) for x in factored) + ")"


def factor_notation_to_integer(s):
    if s == "0":
        return 0
    paren_groups = get_top_level_paren_groups(s)
    print(paren_groups)
    # TODO: finish parsing recursively and converting to int


def get_top_level_paren_groups(s):
    # this strips parens from each group
    groups = []
    current = ""
    paren_count = 0
    for x in s:
        current += x
        if x == "(":
            paren_count += 1
        elif x == ")":
            paren_count -= 1
        if paren_count == 0:
            groups.append(current)
            current = ""
            # if x == ")":
            #     raise ValueError("close paren without preceding open")
            # if x != "(":
            #     groups.append(x)
            #     current = ""
    assert paren_count == 0
    return groups


if __name__ == "__main__":
    n = (2 ** 6) * (3 ** 4) * (5 ** 5)
    # n = (2 ** 16) - 1
    # n = 16 * 3
    # n = 7
    # n = 2
    print(integer_to_factor_notation(n))
    # s = "(((0))(0))"
    s = "(0)"
    print(factor_notation_to_integer(s))

    # TODO: make turtle graphics from the notation of different numbers (treating them as L-Systems)