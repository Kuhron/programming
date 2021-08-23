# estimating normal distribution CDF at extremely low values using Taylor series for erf

import math


def cdf(x, n_terms, mu=0, sigma=1):
    return 1/2 * (1 + erf((x-mu) / (sigma * math.sqrt(2)), n_terms))


def erf(z, n_terms):
    terms = []
    cumulative = 0
    avgs = []
    last_log_term = None
    for i in range(n_terms):  # indexed from zero in the Taylor series on Wikipedia
        term_sign, abs_log_term = erf_taylor_term_log(z, i)
        if last_log_term is None:
            log_diff = None
        else:
            log_diff = abs_log_term - last_log_term
        print(f"estimating erf({z}); term {i+1} has term_sign {term_sign}, abs_log_term {abs_log_term}")
        last_log_term = abs_log_term

        term = term_sign * math.exp(abs_log_term)
        cumulative += term
        terms.append(term)
        avg = cumulative / (i+1)
        avgs.append(avg)
        # sometimes they diverge and alternate so do a 1-2+3-4+... thing (Cesaro summation: arithmetic mean of first N partial sums, lim N->inf)
        print(f"Cesaro approximation at term i={i}: {avg}")
    return avg


def erf_taylor_term_log(z, n):
    # https://en.wikipedia.org/wiki/Error_function#Taylor_series
    c = 2 / math.sqrt(math.pi)
    sign_term = (-1)**n
    # numer = (-1)**n * z**(2*n+1)
    # denom = math.factorial(n) * (2*n + 1)
    # try getting around numerical problems by using logs, I think the main overflow is z**p not n! (for large z)
    # if z is negative, log is not defined, but because (2*n+1) is always odd, the z**p term will have the same sign as z
    sign_z = -1 if z < 0 else 0 if z == 0 else 1
    sign_term *= sign_z  # this way we incorporate the (-1)**n and the z**(p) into one sign variable, then we can take log abs z
    log_numer = math.log(c) + (2*n + 1) * math.log(abs(z))
    log_denom = math.log(math.factorial(n)) + math.log(2*n + 1)
    
    # return c * numer / denom
    log_term_without_sign = log_numer - log_denom
    return (sign_term, log_term_without_sign)


if __name__ == "__main__":
    # x = -345e6
    x = 1
    n_terms = 1000
    print(cdf(x, n_terms))  # this is messed up, gives wrong value for x=1, and Cesaro sequence diverges still for z=-345M
