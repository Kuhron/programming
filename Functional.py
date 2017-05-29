run = lambda fs: fs[0](fs[1]) if len(fs) == 2 else run(fs[:-1])(fs[-1])

combinator = lambda f: lambda *args: f(f, *args)
fibonacci_func = lambda f, n: [1] * (n + 1) if n <= 1 else (lambda x: x + [sum(x[-2:])])(f(f, n - 1))

fs = (
    (combinator),
    (fibonacci_func),
    (10),
)

print(run(fs))