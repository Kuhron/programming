# -*- coding: utf-8 -*-
# attempt to implement some compression algorithms as well as ideas of my own

import sys


def lzw_compress(s):
    # Lempel-Ziv-Welch algorithm
    raise NotImplementedError


def get_longest_prefix(s, t):
    # stolen from https://stackoverflow.com/questions/10355103/finding-the-longest-repeated-substring
    p = ""
    for x, y in zip(s, t):
        if x != y:
            return p
        p += x

    min_str = s if len(s) <= len(t) else t
    assert p == min_str
    return p


def get_longest_repeated_substring(s):
    # stolen from https://stackoverflow.com/questions/10355103/finding-the-longest-repeated-substring
    suffixes = sorted(s[i:] for i in range(len(s)))
    result = ""
    for s1, s2 in zip(suffixes[:-1], suffixes[1:]):
        prefix = get_longest_prefix(s1, s2)
        if len(prefix) > len(result):
            result = prefix
    return result


def char_ints():
    i = ord("0")
    while True:
        yield i
        i += 1


def grammar_compress_one_stage(s, seen_chars):
    substr = get_longest_repeated_substring(s)

    if len(substr) <= 1:
        return s, seen_chars

    try:
        current_char = next(c for c in seen_chars if c not in s)
    except StopIteration:
        current_char = next(x for x in char_ints() if chr(x) not in s)
    new_symbol = chr(current_char)
    seen_chars.add(new_symbol)

    s_compressed = s.replace(substr, new_symbol)
    s_compressed += "\n{}={}".format(new_symbol, substr)

    return s_compressed, seen_chars


def grammar_compress(s):
    # idea based on formal grammars
    # each stage acts on the whole text (probably pretty slow)
    # in each stage, find some long string that is repeated, and replace it with a symbol not in the grammar's keys
    # in the final file, print the grammar for decoding (if this can be eliminated, it would probably save a lot of space)

    seen_chars = set(s)
    while True:
        s_compressed, seen_chars = grammar_compress_one_stage(s, seen_chars)
        if len(s_compressed) >= len(s):
            return s
        s = s_compressed


def evaluate_compression_function(func, s):
    ls = len(s)
    lc = len(func(s))
    r = 100 * (1 - lc / ls)
    print("original len {}, new len {} (decreased by {:.2f}%)".format(ls, lc, r))


if __name__ == "__main__":
    fp = "LoremIpsum.txt"
    # fp = "Compression.py"
    s = open(fp).read().lower()  # unicameral alphabet for now
    func = grammar_compress

    evaluate_compression_function(func, s)
    output_fp = "CompressionOutput.txt"
    with open(output_fp, "wb") as f:
        f.write(func(s).encode("utf-8"))