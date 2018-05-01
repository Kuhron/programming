import random
import string

code = "".join(random.choice(string.ascii_lowercase) for _ in range(3))

short_str = "isolang {}".format(code)
long_str = "https://en.wikipedia.org/wiki/ISO_639:{}".format(code)

print(short_str)
print(long_str)
