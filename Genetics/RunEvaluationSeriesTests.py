import numpy as np
import matplotlib.pyplot as plt
from RawVariation import get_dna, transcribe_dna, flip_bit
from FunctionOfBits import linear_choice_series, signed_log


def run_evaluation_series_test(eval_func):
    dna = get_dna(400)
    xs = []
    while len(dna) > 0:
        x = eval_func(dna)
        print(x)
        xs.append(x)
        dna = transcribe_dna(dna)
    plt.plot(xs)
    plt.savefig("DnaEvaluationSeries.png")
    plt.gcf().clear()


def show_effect_of_bases_test(eval_func):
    dna = get_dna(400)
    # evolve/mutate it a bit so it's not just p=0.5 random choice, want some nontrivial structure
    for i in range(20):
        dna = transcribe_dna(dna)
        if len(dna) == 0:
            dna = get_dna(400)

    xs = []
    dna_val = eval_func(dna)
    for i in range(len(dna)):
        new_dna = flip_bit(dna, i)
        new_dna_val = eval_func(new_dna)
        diff = new_dna_val - dna_val
        xs.append(diff)
        print(f"bit {i} has effect of {diff} ({dna_val} --> {new_dna_val})")
    plt.subplot(2,1,1)
    plt.plot(xs, c="b")
    plt.subplot(2,1,2)
    plt.plot(dna, c="r")
    plt.savefig("EffectsOfBaseFlips.png")
    plt.gcf().clear()


if __name__ == "__main__":
    coefficients = np.random.normal(0,1,4)
   
    def eval_func(dna):
        xs = linear_choice_series(dna, coefficients, modification_function=signed_log)
        # return xs[-1]  # overweights importance of later bits
        # return xs.mean()  # importance moves nicely, correlated with each bit's value
        # return xs.max()  # more sporadic importance of certain bits depending on the dna string, I like this
        # return xs.max() - xs.min()  # also nice
        xmax = xs.max()
        xmin = xs.min()
        xmean = xs.mean()
        # alpha_min_mean_max = (xmean - xmin) / (xmax - xmin)
        # return alpha_min_mean_max
        top_half_minus_bottom_half = (xmax - xmean) - (xmean - xmin)
        return top_half_minus_bottom_half

    run_evaluation_series_test(eval_func)
    show_effect_of_bases_test(eval_func)
