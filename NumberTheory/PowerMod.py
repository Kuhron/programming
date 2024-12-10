import matplotlib.pyplot as plt


def f(p, m, max_n):
    plt.hist([(x**p) % m for x in range(1, max_n+1)], bins=2*m-1)
    plt.show()


while True:
    inp = input("p,m,max_n: ").split(",")
    try:
        p,m,max_n = inp
    except ValueError:
        p,m = inp
        max_n = 10**6
    p = int(p)
    m = int(m)
    max_n = int(max_n)
    f(p, m, max_n)
