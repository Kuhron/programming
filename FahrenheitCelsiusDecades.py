f_to_c = lambda x: (x-32)*5/9
c_to_f = lambda x: x*9/5 + 32

for c in range(-20, 61, 10):
    print(f"{c} C = {int(c_to_f(c))} F")
print()

for f in range(0, 121, 10):
    c = f_to_c(f)
    c_int = round(c)
    diff = c - c_int
    diff_ninths = diff * 9
    assert 0 <= (diff_ninths % 1) <= 1e-6 or 1-1e-6 <= (diff_ninths % 1) < 1, diff_ninths  # float crap
    diff_ninths = round(diff_ninths)
    print(f"{f} F = {c_int} C ({diff_ninths:+}/9)")
