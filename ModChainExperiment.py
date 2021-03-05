# idea: given a list of ints, see how the result of the mod operation changes based on how they are sub-grouped into a syntax tree
# e.g. 4 % 7 % 5 = 4 but 4 % (7 % 5) = 0

import random
import math
import numpy as np
import matplotlib.pyplot as plt


def get_all_syntax_groupings(lst):
    if len(lst) == 0:
        raise ValueError("can't group empty list")
    if len(lst) == 1:
        only_grouping = [lst[0]]  # each grouping is itself a list
        return [only_grouping]
    if len(lst) == 2:
        only_grouping = [lst[0], lst[1]]  # I know I could do return [lst] for len of 1 or 2, but this makes it clearer to me what is going on; len-2 list will be a syntax tree with two children on the root
        return [only_grouping]
    else:
        # for each adjacent pair you could choose, group that pair into a sub-list and then re-run on the new list which now has one fewer len
        groupings = []
        for i in range(len(lst)-1):
            pair = lst[i:i+2]
            new_lst = lst[:i] + [pair] + lst[i+2:]
            assert len(new_lst) == len(lst) - 1
            groupings += get_all_syntax_groupings(new_lst)
        assert len(groupings) == math.factorial(len(lst)-1)
        return groupings


def evaluate_grouping(g, func):
    # func needs to have 2 args
    x, y = g
    if type(x) is list:
        x = evaluate_grouping(x, func)
    if type(y) is list:
        y = evaluate_grouping(y, func)
    return func(x, y)


def plot_slice(func, n_dims):
    assert n_dims >= 2, "too few dimensions for slice"
    arr = [random.randint(1,100) for i in range(n_dims)]
    # select two indices in the array for the variables
    x_index, y_index = random.sample(range(n_dims), 2)
    arr[x_index] = "x"
    arr[y_index] = "y"
    xs = list(range(1,101))
    ys = list(range(1,101))
    zs = [[None for y in ys] for x in xs]
    for x_i, x in enumerate(xs):
        for y_i, y in enumerate(ys):
            this_arr = [a for a in arr]
            this_arr[x_index] = x
            this_arr[y_index] = y
            z = get_average_syntax_tree_value(this_arr, func)
            zs[x_i][y_i] = z
    plt.imshow(zs)
    plt.colorbar()
    plt.title(arr)
    plt.show()


def get_average_syntax_tree_value(arr, func):        
    groupings = get_all_syntax_groupings(arr)
    values = [evaluate_grouping(g, func) for g in groupings]
    has_nan = np.nan in values  # this does work even though nan is not equal to itself; maybe it's checking with `is` instead of `==`
    if has_nan:
        values_without_nan = [x for x in values if not np.isnan(x)]
    else:
        values_without_nan = values
    avg_value = np.mean(values_without_nan)  # do this before putting nan back in; also, do count multiplicity
    return avg_value


def report_stats(int_lst, func):
    groupings = get_all_syntax_groupings(lst)
    values = [evaluate_grouping(g, func) for g in groupings]
    x_max = len(values)

    has_nan = np.nan in values  # this does work even though nan is not equal to itself; maybe it's checking with `is` instead of `==`
    if has_nan:
        values_without_nan = [x for x in values if not np.isnan(x)]
    else:
        values_without_nan = values
    avg_value = np.mean(values_without_nan)  # do this before putting nan back in; also, do count multiplicity

    unique_values = set(values_without_nan)
    if has_nan:
        unique_values.add(np.nan)  # set(lst) will keep all nan's since it uses `==` instead of `is`
    n_values = len(unique_values)
    unique_values = sorted(unique_values)

    print("integers: {}".format(int_lst))
    print("{} unique values found: {}".format(n_values, unique_values))
    print("average value: {}".format(avg_value))

    # plot the values by syntax tree number (since the trees are well-ordered w.r.t. each other)
    xs = list(range(x_max))
    ys = values
    assert len(xs) == len(ys) == x_max

    # convolve to smooth it out
    mu = 0
    sigma = 2  # can adjust this to make more/less smooth (widening the kernel with larger sigma makes smoother function)
    assert sigma > 0
    norm_pdf = lambda x: 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2)
    norm_pdf_inverse = lambda p: mu + sigma * np.sqrt(-2*np.log(p * (sigma*np.sqrt(2*np.pi))))  # inverted norm_pdf manually myself, could have mistakes

    # test_xs = np.linspace(-5,5,100)
    # test_ys = norm_pdf(test_xs)
    # see if it looks correct
    # plt.plot(test_xs, test_ys)
    # plt.show()
    
    # convolution doesn't work on unevenly spaced data (if there are nans)
    # kernel = test_ys / sum(test_ys)  # normalize
    # convolution = np.convolve(ys, kernel)

    # try this: weighted average of samples, weighted by norm_pdf of their distance
    # ignore samples that are so far away that the pdf is less than 1/1000
    # min_pdf_value = 1/1000
    # max_displacement = norm_pdf_inverse(1/1000)
    # assert max_displacement > 0, "problem in norm inverse"
    # assert abs(norm_pdf(max_displacement) - min_pdf_value) < 1e-6, "problem in norm inverse"
    # on second thought, don't have min value, since it will make there be no estimation at a point that is more than this displacement from any sample

    # wrap around, so the result function is a continuous periodic signal
    estimation_xs = np.arange(0,1,1/x_max)  # put the domain just on [0,1]
    # but don't want 0 and 1 to overlap when wrapping around, so exclude endpoint at 1
    assert 1 not in estimation_xs and estimation_xs[-1] == 1-1/x_max
    assert len(estimation_xs) == x_max
    estimation_ys = np.zeros((x_max,))
    total_weights = np.zeros((x_max,))
    normalized_x_max = 1
    for x_i, x in enumerate(xs):
        # here we still get x from the original list, not normalized
        distances_to_observations = [min(abs(x-(obs-x_max)), abs(x-obs), abs(obs+x_max - x)) for obs in xs]
        observation_weights = [norm_pdf(d) for d in distances_to_observations]
        for y, w in zip(ys, observation_weights):
            if np.isnan(y):
                continue
            estimation_ys[x_i] += w*y
            total_weights[x_i] += w
    # now need to normalize the ys by the weight at each point
    estimation_ys /= total_weights

    # now get fourier transform of the signal made by repeating this smoothed function
    signal_one_repetition = estimation_ys - np.mean(estimation_ys)
    n_repetitions = 100
    signal = np.tile(signal_one_repetition, n_repetitions)  # try to remove edge effects
    fft = np.fft.fft(signal)
    signal_amplitude = abs(fft)  # python has abs of complex number built in
    # need to scale the fft's frequency values back down by n_repetitions (e.g. it will have a spike at 100 for a 1-per-cycle wave when n_repetitions is 100)
    freq_xs = np.arange(0, len(fft)/n_repetitions, 1/n_repetitions)
    # it's always a mirror image for reasons I don't understand, so just look at the first half of the spectrum
    freq_xs = freq_xs[:math.ceil(len(freq_xs)/2)]
    signal_amplitude = signal_amplitude[:math.ceil(len(signal_amplitude)/2)]

    # plot the points and the convolution together
    plt.subplot(2,1,1)
    plt.title("data and smoothed function")
    plt.scatter(estimation_xs, ys, c="blue")  # now squish the x domain but keep original ys
    plt.plot(estimation_xs, estimation_ys, c="red")
    plt.subplot(2,1,2)
    plt.title("spectrum of smoothed function")
    plt.plot(freq_xs, signal_amplitude)
    plt.show()            
        
    

if __name__ == "__main__":
    func = lambda x, y: x % y if y != 0 else np.nan
    lst = [random.randint(1, 100) for i in range(6)]
    report_stats(lst, func)

    plot_slice(func, n_dims=5)
