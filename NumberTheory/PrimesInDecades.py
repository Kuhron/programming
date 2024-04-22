import matplotlib.pyplot as plt

from Eratosthenes import get_all_primes


def get_primes_in_decades(dec_max):
    # dec_max is the last decade we want (p // 10), inclusive
    # e.g. dec_max=3 will give highest prime of 37
    d = {dec: [] for dec in range(dec_max + 1)}
    ps = get_all_primes()
    for p in ps:
        dec = p // 10
        if dec > dec_max:
            break
        d[dec].append(p % 10)
    return d


def get_binary_string_for_prime_ones_digits(ones):
    allowed = [1,3,7,9]
    if any(x not in allowed for x in ones):
        # should only happen for 2 and 5
        return "ill_defined"
    s = ""
    for x in allowed:
        s += str(int(x in ones))
    return s


def get_binary_number_from_string(s):
    n = 0
    for i, x in enumerate(s):
        if x not in ["0", "1"]:
            return "ill_defined"
            # raise ValueError(f"bad char {x!r} in string {s!r}")
        p = len(s) - i - 1
        if x == "1":
            n += 2**p
    return n


if __name__ == "__main__":
    dec_max = 99
    d = get_primes_in_decades(dec_max)

    xs = []
    ys = []
    with open("decades.txt", "w") as f:
        for dec in range(dec_max):
            ones = d[dec]
            n_ps = len(ones)
            bin_s = get_binary_string_for_prime_ones_digits(ones)
            bin_n = get_binary_number_from_string(bin_s[::-1])
            if type(bin_n) is int:
                xs.append(dec)
                ys.append(bin_n)

            prime_s = ",".join(str(10*dec + x) for x in ones)
            s = f"dec={dec}\tbin={bin_s}={bin_n}\tcount={n_ps}\tps={prime_s}\n"
            f.write(s)

    plt.scatter(xs, ys)
    plt.show()
