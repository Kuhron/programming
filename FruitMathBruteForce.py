from RationalNumberGenerator import rational_number_tuple_generator


def f(a,b,c):
    return a/(b+c) + b/(a+c) + c/(a+b) - 4  # the hard elliptic curve original problem
    # return a*b**2 + b*c**2 + c*a**2 - 1  # test case
    # return 2*a**2 + 4*b**2 + 6*a*b - b*c - 7*a*c - 11  # test case


def f_rational_tuple(pq_a, pq_b, pq_c):
    pa, qa = pq_a
    pb, qb = pq_b
    pc, qc = pq_c
    return f(pa/qa, pb/qb, pc/qc)


def is_solution3(a,b,c):
    return f(a,b,c) == 0


def is_solution2(a,b):
    c = 1
    return f(a,b,c) == 0


def is_solution2_rational_tuple(pq_a, pq_b):
    pq_c = (1,1)
    return f_rational_tuple(pq_a, pq_b, pq_c) == 0


def sub_gen(g_func, n):
    g = g_func()
    return [next(g) for i in range(n)]


def brute_force():
    diag_i = 1
    while True:
        if diag_i % 100 == 0:
            print("checking diagonal {}".format(diag_i))
        lst_a = sub_gen(rational_number_tuple_generator, diag_i)
        lst_b = sub_gen(rational_number_tuple_generator, diag_i)
        for pq_a, pq_b in zip(lst_a, lst_b[::-1]):
            # print("checking {} {}".format(pq_a, pq_b))
            if is_solution2_rational_tuple(pq_a, pq_b):
                print("solution found! {} {}".format(pq_a, pq_b))
                return
        diag_i += 1


if __name__ == "__main__":
    brute_force()
