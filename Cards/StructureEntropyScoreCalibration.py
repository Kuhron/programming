# game could be called "Coset" since that's the structure type you're trying to build,
# but I like the role of Entropy in the title so idk yet

data_fp = "StructureEntropyGameData.txt"

with open(data_fp) as f:
    lines = f.readlines()

tups = []
for line in lines:
    if line.startswith("#"):
        continue
    n, k, x, result_type = line.split(",")
    n = int(n)
    k = int(k)
    x = int(x)
    tups.append((n,k,x))

def get_score_single_tup(tup, reward_func, held_penalty_func, thrown_penalty_func):
    # the penalty funcs should be abs, since they are subtracted here rather than added
    n,k,x = tup
    if n > 1:
        assert n == k
        return reward_func(n) - thrown_penalty_func(x)
    else:
        # also do this for coset of size one since it's trivial, treat this as a loss
        return -held_penalty_func(k) - thrown_penalty_func(x)

def get_score(tups, reward_func, held_penalty_func, thrown_penalty_func):
    return sum(get_score_single_tup(tup, reward_func, held_penalty_func, thrown_penalty_func) for tup in tups)

possible_numbers_of_numbers_in_a_suit = [1, 2, 3, 4, 6, 12]
possible_numbers_of_suits = [1, 2, 4]
possible_orders_of_subgroups = sorted(set(x*y for x in possible_numbers_of_numbers_in_a_suit for y in possible_numbers_of_suits))
print("possible orders:", possible_orders_of_subgroups)

order_index = lambda n: {1:0, 2:1, 3:2, 4:3, 6:4, 8:5, 12:6, 16:7, 24:8, 48:9}[n] 
reward_func = lambda n: 0 if n == 1 else 10*n
held_penalty_func = lambda k: k*2
thrown_penalty_func = lambda x: x

for tup in tups:
    print(tup, get_score_single_tup(tup, reward_func, held_penalty_func, thrown_penalty_func))
print("Total score:", get_score(tups, reward_func, held_penalty_func, thrown_penalty_func))
