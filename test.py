#!/bin/python3

import time


def get_inputs():
    n,k = input().strip().split(' ')
    n,k = [int(n),int(k)]
    x = [int(x_temp) for x_temp in input().strip().split(' ')]
    assert len(x) == n
    return x, k


# def transform_x(array):
#     return [x - array[0] for x in array]


def get_first_location_greedy(array, k):
    if len(array) == 1:
        return array[0]
    result = None
    a0 = array[0]
    for x in array:
        if x - a0 <= k:
            result = x
        else:
            break
    return result


def get_all_locations(array, k):
    # result = []
    len_result = 0
    array = sorted(array)
    loop_limit = 10**5 + 1
    loop_limiter = 0
    while len(array) > 0:
        loop_limiter += 1
        if loop_limiter > loop_limit:
            raise
        # array = transform_x(array)
        location = get_first_location_greedy(array, k)
        assert location is not None, "error on array {}".format(array)
        # result.append(location)
        len_result += 1
        array_snipped = False
        for i, x in enumerate(array):
            if x > location + k:
                array = array[i:]
                array_snipped = True
                break
        if not array_snipped:
            array = []
    # return result
    return len_result


# t0 = time.time()
xs, k = get_inputs()
print(get_all_locations(xs, k))
# print(time.time() - t0)