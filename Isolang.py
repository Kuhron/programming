import random
import string

print("isolang " + "".join(random.choice(string.ascii_lowercase) for _ in range(3)))