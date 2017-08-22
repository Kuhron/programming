# attempt to implement some compression algorithms as well as ideas of my own


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


def ints():
    i = 0
    while True:
        yield i
        i += 1


def grammar_compress_one_stage(s, grammar):
    # current_char = max(ord(c) for c in s) + 1
    current_char = next(x for x in ints() if chr(x) not in s)
    new_symbol = chr(current_char)
    substr = get_longest_repeated_substring(s)

    # new_grammar = grammar + {new_symbol: substr}  # wrong syntax, and may I just say it's annoying that d.update(...) mutates and returns None
    # new_grammar = {new_symbol: substr, **grammar}
    # f this
    new_grammar = grammar.copy()
    new_grammar.update({new_symbol: substr})

    s_compressed = s.replace(substr, new_symbol)

    # # not necessary if each stage is a separate function call
    # while current_char in s_compressed:
    #     current_char += 1

    return s_compressed, new_grammar


def grammar_compress(s):
    # idea based on formal grammars
    # each stage acts on the whole text (probably pretty slow)
    # in each stage, find some long string that is repeated, and replace it with a symbol not in the grammar's keys
    # in the final file, print the grammar for decoding (if this can be eliminated, it would probably save a lot of space)

    grammar = {}
    length = len(s) + len(repr(grammar))
    while True:
        s_compressed, new_grammar = grammar_compress_one_stage(s, grammar)
        if len(s_compressed) + len(repr(new_grammar)) >= length:
            return s + "\n" + repr(grammar)
        s, grammar = s_compressed, new_grammar


def evaluate_compression_function(func, s):
    return 1 - len(func(s)) / len(s)


if __name__ == "__main__":
    fp = "LoremIpsum.txt"
    # fp = "Compression.py"
    s = open(fp).read().lower()  # unicameral alphabet for now
    func = grammar_compress
    # print(evaluate_compression_function(func, s))
    print(func(s))