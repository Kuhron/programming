s = "FizzBuzz"
lst = map(lambda i: max((s[:4] * (i % 3 == 0)) + (s[4:] * (i % 5 == 0)), str(i)), range(1, 31))
print("\n".join(lst))
