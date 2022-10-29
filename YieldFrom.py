def get_ns():
    for i in range(10):
        yield i


def yield_from_in_loop(g):
    for i in range(11, 20):
        yield i
        yield from g  # this yields everything in g before continuing the loop


def yield_from_bare(g):
    yield from g


def capture_yield(g):
    for x in g:
        # capture yielded value; parentheses make yield act as an expression
        x2 = (yield x**2)  # we yield x**2 and also assign it?
        print(f"{x2=}")


ns = get_ns()
# g = yield_from_in_loop(ns)
# g = yield_from_bare(ns)
g = capture_yield(ns)
for y in g:
    print(f"{y=}")
