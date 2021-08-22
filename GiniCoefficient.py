import random
import numpy as np
import matplotlib.pyplot as plt


def get_incomes(n):
    a = random.uniform(1.1, 5)  # 1 gives infinite mean and variance so how bout let's don't
    pareto = np.random.pareto(a, (n//4,))

    mu = random.uniform(0, max(pareto))
    sigma = random.uniform(0, max(pareto)/2)
    normal = abs(np.random.normal(mu, sigma, (3*n//4,)))

    res = np.concatenate([pareto, normal])
    m = sum(random.choice(res) for i in range(4)) / 4 + random.choice(res)  # enforce some minimum value
    res += m
    assert res.ndim == 1, "concat didn't work"
    return res


def get_incomes_more_mixed(n, k):
    # k rounds of n
    res = None
    for i in range(k):
        a = get_incomes(n)
        if res is None:
            res = a
        else:
            res = np.concatenate([res, a])
        assert res.ndim == 1
    return res


def get_perfectly_fair_incomes():
    return np.ones(10000)


def get_perfectly_unfair_incomes():
    return [1] + [0]*9999


def get_lorenz_curve(incomes):
    # x is proportion of the population
    # y is income share earned by the bottom x of the population
    incomes = sorted(incomes)
    n = len(incomes)
    ys = {}
    cumulative_income = 0
    total_income = sum(incomes)
    for person_i, inc in enumerate(incomes):
        cumulative_income += inc
        cumulative_population_proportion = (person_i + 1) / n
        x = cumulative_population_proportion
        income_proportion = cumulative_income / total_income
        y = income_proportion
        ys[x] = y
    return ys


def get_gini_coefficient(incomes):
    incomes = sorted(incomes)
    lorenz_ys = get_lorenz_curve(incomes)
    # the line of equality is just y = x (proportion of the population has the same proportion of income below it)
    total_area = 0
    area_under_curve = 0
    for x, y in lorenz_ys.items():
        total_area += x  # since y = x
        area_under_curve += y
        assert y <= x, "can't have income going above the y=x line"
    area_above_curve = total_area - area_under_curve
    g = area_above_curve / total_area  # 0 for perfect equality, 1 for perfect tyranny
    return g


if __name__ == "__main__":
    incomes = get_incomes_more_mixed(n=1000, k=20)
    # incomes = get_perfectly_fair_incomes()  # expect g == 0
    # incomes = get_perfectly_unfair_incomes()  # expect g close to 1
    plt.hist(incomes, bins=100)
    plt.gca().set_yscale("log")
    plt.title(get_gini_coefficient(incomes))
    plt.savefig("GiniPlot.png")
