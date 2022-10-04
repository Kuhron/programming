# fit 11th-degree polynomial to get number of days in a month
# do one for common year and one for leap year

import numpy as np

cdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ldays = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
x = list(range(1, 13))

ccoeff = np.polyfit(x, cdays, 11)
lcoeff = np.polyfit(x, ldays, 11)

cpoly = np.polynomial.Polynomial(ccoeff[::-1])
lpoly = np.polynomial.Polynomial(lcoeff[::-1])

np.polynomial.set_default_printstyle('ascii')
for i in x:
    print(i, cpoly(i), lpoly(i))

print(cpoly)
print(lpoly)
