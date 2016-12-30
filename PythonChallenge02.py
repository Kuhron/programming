#Originally written for Python 2.x or whatever Codecademy taught me
mess = ""
entry = input("input:\n")
while entry != "":
    mess = mess + entry
    entry = input("")

counts = {}
for i in mess:
    if i not in counts:
        counts[i] = 1
    else:
        counts[i] += 1
print(counts)

cut = {}
for u in counts:
    if counts[u] <= 10:
        cut[u] = counts[u]
    else:
        pass
print(cut)

clean = ""
for i in mess:
    if i in cut:
        clean = clean + i
print(clean)
