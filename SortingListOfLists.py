import random

lsts = []
for i in range(100):
    lst = []
    expected_length = 10
    while random.random() < 1 - 1/expected_length:
        lst.append(random.randint(0, 9))
    lsts.append(lst)

for lst in sorted(lsts):
    print(lst)

