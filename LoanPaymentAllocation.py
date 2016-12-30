# goal: minimize total amount paid over all time

# HYPOTHESIS 1: ALWAYS PAY DOWN THE HIGHEST INTEREST RATE FIRST
# HYPOTHESIS 2: PAY DOWN THE LOAN WITH THE HIGHEST INTEREST ACCRUING IN THE
    # NEXT PERIOD FIRST
    # (e.g. $1 at 100% (+$1) vs. $1000 at 1% (+$10)
# other hypotheses?

# all matrices are read downward in time
# i.e., each row is one period and each column is a loan
# vectors need not be expressed as matrices, but be careful with interpretation

num_loans = 4
num_periods = 5

# income vector, allocatable over different loans
# VECTOR OVER TIMES
incomes = [100, 100, 100, 100, 100]

# initial principals of all loans
# VECTOR OVER LOANS
initial_principals = [50, 40, 100, 20]

# interest rates of all loans, constant in time
# VECTOR OVER LOANS
rates = [0.05, 0.06, 0.07, 0.08]

def all_zero(lst):
    for i in lst:
        if i != 0:
            return False
    return True

def allocate_highest_rate(m, principals, rs):
    """
    Allocates an income amount over the set of loans based solely on rate.
    """
    # if two loans have the same interest rate, treat them as basically
    # the same loan, so just choosing the first one you find is fine
    # this is the behavior of list.index() to my knowledge
    result = [0 for u in range(len(rs))]
    if all_zero(principals):
        return result
    index = rs.index(max(rs))
    if m > principals[index]*(1+rs[index]):
        print("YOU"RE NOT DONE")
    result[index] += m
    return result

def allocate_highest_accrual(m, principals, rs):
    """
    Allocates an income amount over the set of loans based on the
    next interest accrual, assuming unpaid interest capitalizes.
    NOTE: the current (not initial) principals should be used.
    """
    if all_zero(principals):
        return [0 for u in range(len(rs))]
    accruals = [rs[u] * principals[u] for u in range(len(rs))]
    index = accruals.index(max(accruals))
    return [0 for u in range(index)] + [m] + [0 for u in range(index+1, len(rs))]

def simulate(ms, ips, irs, rule):
    """
    Takes incomes, initial principals, rates, and a rule argument
    ("highest_rate" or "highest_accrual")
    Returns a matrix of the payments to the loans over all periods.
    """
    T = len(ms)
    result = []
    ps = ips[:] # copy the initial principals, will be changed by payments
    rs = irs[:]
    for t in range(T):
        if rule == "highest_rate":
            payment_vector = allocate_highest_rate(ms[t], rs)
        elif rule == "highest_accrual":
            payment_vector = allocate_highest_accrual(ms[t], ps, rs)
        result.append(payment_vector)
        ps = [ps[u]-payment_vector[u] for u in range(len(ps))]
    return result

if __name__ == "__main__":
    if len(incomes) != num_periods:
        print("vector \"incomes\" of wrong length")
    if len(initial_principals) != num_loans:
        print("vector \"initial_principals\" of wrong length")
    if len(rates) != num_loans:
        print("vector \"rates\" of wrong length")

