import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# process that changes each of its own parameters as a step before generating a new value
# each "type" of parameter (e.g., the alpha of an EMA) has only ONE instance and is used for all such purposes

# expect to see things change over time, including:
# drift, variance, kurtosis, autocorrelation
# attempt to be as non-stationary, unpredictable, and regime-changing as possible

# always get parameters by calling functions in the global scope on the last series value, which is the real-valued seed

# the only actual random function used, for seeding; everything else is deterministic based on x
r = lambda a, b: np.random.uniform(a, b)

# functions to transform a real number x into the desired type of parameter, in many cases confined to an interval

def exp(x):
    while True:
        val = np.exp(x)
        if not np.isnan(val):
            return val
        else:
            x = np.sqrt(x)

def real_to_interval(x, a, b):
    sigmoid_0_1 = 1 / (1 + exp(-1 * x))
    return a + sigmoid_0_1 * (b - a)

def hash_to_interval(x, a, b):
    x = x % 1
    x = 1 / x
    x += (-1 if x >= 0 else 1)
    return real_to_interval(x, a, b)
    
def get_ema_alpha(x):
    return real_to_interval(x, 0, 1)

def get_probability(x):
    return real_to_interval(x, 0, 1)

def get_sign(x):
    sin_value = np.sin(translate(dilate(x)))
    return 1 if sin_value >= 0 else -1

def get_step_amount(x):
    return stats.norm.cdf(x, loc=get_normal_mu(x), scale=get_normal_sigma(x))

def get_scale_amount(x):
    return x * abs(np.sin(x))

def get_normal_mu(x):
    return real_to_interval(x, -1*get_scale_amount(x), get_scale_amount(x))

def get_normal_sigma(x):
    return get_scale_amount(exp(x))

def get_boolean(x):
    # return real_to_interval(x, 0, 1) < real_to_interval(x, 1, 0)
    return np.random.choice([True, False])


# transformation of a number, done by a changing series of basic operations, which may themselves take params

def translate(x):
    return x + get_step_amount(x)

def dilate(x):
    return x * get_scale_amount(x)

def flip(x):
    return x * get_sign(x)

transformation_funcs = [translate, dilate, flip]

def transform(x, transformations):
    transformations = modify_transformations(x, transformations)
    for func in transformations:
        x = func(x)
    return x, transformations

def get_random_transformation(x):
    index = int(x % len(transformation_funcs))
    return transformation_funcs[index]

def modify_transformations(x, transformations):
    if len(transformations) == 0:
        transformations.append(get_random_transformation(x))
    new = []
    for f in transformations:
        if get_boolean(x):
            new.append(f)
        if get_boolean(x):
            new.append(get_random_transformation(x))
    return new


# time series generator

def get_time_series():
    x = r(-1, 1)
    transformations = []
    while True:
        print(x)
        print(len(transformations))
        x, transformations = transform(x, transformations)
        yield x


if __name__ == "__main__":
    xs = [i for i in range(20)]
    gen = get_time_series()
    ys = [next(gen) for x in xs]
    plt.plot(xs, ys)
    plt.show()
