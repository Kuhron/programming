collatz = lambda n: n/2 if n % 2 == 0 else 3*n+1

def chain(n):
    res = [n]
    while n != 1:
        n = collatz(n)
        res.append(n)
    return res

max_chain_len = 0
ans = None

for n in range(1, 1000000):
    if n % 10000 == 0: print(n)
    chain_len = len(chain(n))
    if chain_len > max_chain_len:
        ans = n
        max_chain_len = chain_len

print(ans)
