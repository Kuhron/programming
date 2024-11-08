from typing import List
import matplotlib.pyplot as plt
import itertools
import time
import numpy as np

# ----

import random


def get_random_nums_and_target(n):
    a = [None for i in range(n)]
    t = random.randrange(-100, 101)

    target_indices = random.sample(range(n), 2)
    t0 = random.randrange(-250, 251)
    t1 = t - t0
    ti0, ti1 = target_indices
    a[ti0] = t0
    a[ti1] = t1

    forbidden = {t-t0, t-t1}  # just t0 and t1 but this is more transparent for why
    for i in range(n):
        if i == ti0 or i == ti1:
            # already did this one when assigning the target pair
            continue
        while True:
            x = random.randrange(-250, 251)
            if x not in forbidden:
                break
        forbidden.add(t-x)
        a[i] = x

    # assert sum(x+y == t for x,y in itertools.combinations(a,2)) == 1
    return a, t


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        a = nums

        val_to_indices = {}
        for i, x in enumerate(a):
            if x not in val_to_indices:
                val_to_indices[x] = []
            val_to_indices[x].append(i)

        a = sorted(a)
        t = target
        n = len(a)
        # diffs = [a[i+1] - a[i] for i in range(len(a) - 1)]
        mid_index = n//2
        j = mid_index
        i = j - 1
        # start at i,j (middle of the list) and add/subtract as needed?
        # trying to get less than O(n^2)
        seen = set()
        while True:
            assert 0 <= i < j < n, f"require 0 <= {i} < {j} < {n}"
            s = a[i] + a[j]
            seen.add((i,j))
            # print(f"a[{i}] = {a[i]}; a[{j}] = {a[j]}; {s = }; {t = }")
            # d_i_up = diffs[i] if i < n - 1 else None
            # d_i_down = diffs[i-1] if i > 0 else None
            # d_j_up = diffs[j] if j < n - 1 else None
            # d_j_down = diffs[j-1] if j > 0 else None
            if s == t:
                # print("target met")
                if a[i] == a[j]:
                    return val_to_indices[a[i]]
                else:
                    return val_to_indices[a[i]] + val_to_indices[a[j]]
            else:
                i_j_clash = i == j-1
                i_low = i == 0
                j_high = j == n-1

                if s < t:
                    # need to go bigger
                    can_increment_i = not i_j_clash
                    can_increment_j = not j_high
                    if not (can_increment_i or can_increment_j):
                        raise RuntimeError("can't increment!")
                    if not can_increment_i:
                        j += 1
                    elif not can_increment_j:
                        i += 1
                    elif random.random() < 0.5:
                        i += 1
                    else:
                        j += 1
                else:
                    # need to go smaller
                    can_decrement_i = not i_low
                    can_decrement_j = not i_j_clash
                    if not (can_decrement_i or can_decrement_j):
                        raise RuntimeError("can't decrement!")
                    if not can_decrement_i:
                        j -= 1
                    elif not can_decrement_j:
                        i -= 1
                    elif random.random() < 0.5:
                        i -= 1
                    else:
                        j -= 1


if __name__ == "__main__":
    ns = range(10, 1001, 50)
    test_cases = {n: [get_random_nums_and_target(n) for i in range(100)] for n in ns}
    times = {n: [] for n in ns}
    for n in ns:
        print(n)
        for nums, target in test_cases[n]:
            t0 = time.perf_counter()
            sol = Solution().twoSum(nums, target)
            dt = time.perf_counter() - t0
            times[n].append(dt)
        del test_cases[n]
    avg_times = np.array([np.mean(times[n]) for n in ns])
    std_times = np.array([np.std(times[n]) for n in ns])
    sqrt_times = avg_times ** 0.5
    plt.plot(ns, avg_times)
    plt.plot(ns, avg_times + std_times)
    plt.plot(ns, avg_times - std_times)
    plt.show()

