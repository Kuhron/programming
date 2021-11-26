import sympy as sp
sp.init_printing(use_unicode=True)

b, c, m = sp.symbols("b c m")
get_next_numer = lambda numer, denom: sp.expand(b*numer + m*denom)
get_next_denom = lambda numer, denom: sp.expand(numer)

a0_numer = c
a0_denom = 1

numer = a0_numer
denom = a0_denom
numers = [a0_numer]
denoms = [a0_denom]

factor_to_period = {}
period_to_factor = {}

for i in range(1, 40):
    new_numer = get_next_numer(numer, denom)
    new_denom = get_next_denom(numer, denom)
    numer = new_numer
    denom = new_denom
    numers.append(numer)
    denoms.append(denom)
    # print(i, numer/denom)

    # set the term equal to c to get cycle of this period length
    # this results in c*denom - numer = 0 (with convention of wanting the factor of c**2 - bc - m rather than its negative)
    # and we factor the left hand side
    period_1_term = c**2 - b*c - m
    polynomial = sp.expand(c * denom - numer)
    # print("polynomial:", polynomial)
    factored = sp.factor(polynomial / period_1_term)
    constant_coefficient, factors = sp.factor_list(polynomial)
    # print("factors", factors)
    assert (period_1_term, 1) in factors or (-1*period_1_term, 1) in factors
    periods_and_factors_found = []
    for factor, power in factors:
        if power != 1:
            raise Exception(power)
        if factor in factor_to_period:
            period = factor_to_period[factor]
            assert i % period == 0, f"{period} must divide {i} for factor {factor}"
        else:
            factor_to_period[factor] = i
            assert i not in period_to_factor
            period_to_factor[i] = factor

        periods_and_factors_found.append([factor_to_period[factor], factor])
    divisors_needed = sp.divisors(i)
    for period, factor in periods_and_factors_found:
        divisors_needed.remove(period)
        if period == i:
            print(f"unique polynomial for period {i} is {factor}")
        else:
            pass
            # print(f"sub-period {period} is represented by factor {factor}")
    assert len(divisors_needed) == 0, f"period {i} missing divisors {divisors_needed}"
    print()

print("prime factor polynomials for each period:")
for period, factor in sorted(period_to_factor.items()):
    print(period, factor)
