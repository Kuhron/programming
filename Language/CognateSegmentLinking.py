# naive method to detect correspondences between (sequences of) segments

import random
import numpy as np
import matplotlib.pyplot as plt
import functools
import itertools


CONSONANTS = list("mnɲŋptckqbdɟgfvθðszʃʒxɣhʔjwlrɭʈɖɳɽɢʕħ")
VOWELS = list("iyɨɯueøəɤoɛœʌɔæaɑ")


def get_proto_words(n):
    res = set()

    distribution_power = 4
    c_dist = np.random.uniform(0, 1, (len(CONSONANTS),)) ** distribution_power
    v_dist = np.random.uniform(0, 1, (len(VOWELS),)) ** distribution_power
    c_dist /= c_dist.sum()
    v_dist /= v_dist.sum()

    while len(res) < n:
        res.add(get_proto_word(c_dist, v_dist))

    return list(res)


def get_proto_word(c_dist, v_dist):
    n_syllables = random.randint(1, 4)
    return "".join(get_proto_syllable(c_dist, v_dist) for i in range(n_syllables))


def get_proto_syllable(c_dist, v_dist):
    return get_c(c_dist) + get_v(v_dist)


def get_c(c_dist):
    return np.random.choice(CONSONANTS, p=c_dist)


def get_v(v_dist):
    return np.random.choice(VOWELS, p=v_dist)


def mutate_words(words, n_steps):
    sound_changes = []
    for step_i in range(n_steps):
        affect_c_or_v = "c" if random.random() < 0.5 else "v"
        options = CONSONANTS if affect_c_or_v == "c" else VOWELS

        while True:
            inp_sample_word = random.choice(words)
            new_options = [x for x in inp_sample_word if x in options]
            if len(new_options) == 0:
                continue
            inp = random.choice(new_options)
            break

        deletion = random.random() < 0.3
        outp = "" if deletion else random.choice([x for x in options if x != inp])
        epenth = random.choice(VOWELS)  # epenthesis will be added after first consonant of any vowelless word
        # deletion change should not turn any word into empty string, these are immune
        sound_change_probability = random.uniform(0.7, 1)  # sporadicity

        new_words = []
        for w in words:
            new_w = mutate_word_single_change(w, inp, outp, epenth, sound_change_probability)
            new_words.append(new_w)
        words = new_words

        sc = (inp, outp, epenth, sound_change_probability)
        sound_changes.append(sc)

    return words, sound_changes


def mutate_word_single_change(w, inp, outp, epenth, sound_change_probability):
    if random.random() < (1 - sound_change_probability):
        return w  # change didn't apply
    would_be_result = w.replace(inp, outp)
    if len(would_be_result) == 0:
        return w  # word is immune since it would be deleted

    is_consonant_array = [x in CONSONANTS for x in would_be_result]
    if all(is_consonant_array):
        assert len(is_consonant_array) > 0  # I'm paranoid about empty conjunction being True even though I already have a condition for len 0
        # put the epenthetic vowel after the first consonant only
        res = would_be_result[0] + epenth + would_be_result[1:]
    else:
        res = would_be_result

    return res


def count_correspondences(l1_words, l2_words, include_zeros=True):
    # assumes the words are already cognate-aligned
    # for each way to partition each word, align those partitions and draw every possible linking between them that preserves linear order
    # fill any gaps with zero

    # make sparse dictionary
    counts = {}
    for w1, w2 in zip(l1_words, l2_words):
        new_counts = count_correspondences_one_pair(w1, w2, include_zeros)
        for (a, b), c in new_counts.items():
            if (a, b) not in counts:
                counts[(a, b)] = 0
            counts[(a, b)] += c

    return counts


def count_correspondences_one_pair(w1, w2, include_zeros=True):
    partitions_1 = get_partitions(w1)
    partitions_2 = get_partitions(w2)
    counts = {}
    for p1 in partitions_1:
        for p2 in partitions_2:
            # link up the partitions in every possible way
            # FIXME: double counting occurs when the same linking occurs but there are different partitions among its zeros
            # - e.g. kwaru vs kro, one partition has (k:0)(w:0)(a:k)(r:r)(u:o) and another has (kw:0)(a:k)(r:r)(u:o)
            # FIXME hidden nulls not counted, e.g. *wak -> (L1) ak; (L2) wa; then we'd align (a:w)(k:a) when it's really (0:w)(a:a)(k:0)
            # - or more extreme cases where no aligned segments are retained in both languages, e.g. *wari > (L1) wa; (L2) ri
            len1 = len(p1)
            len2 = len(p2)
            max_len = max(len1, len2)
            min_len = min(len1, len2)
            longer_partition_number = 1 if len1 == max_len else 2  # if they're equal, it doesn't matter which one we treat as longer for the sake of linking indices between the partitions
            linking_indices = list(itertools.combinations(range(max_len), min_len))
            # we link from the shorter partition to these indices in the longer partition
            longer_partition = p1 if longer_partition_number == 1 else p2
            shorter_partition = p2 if longer_partition_number == 1 else p1
            for linking in linking_indices:
                unused_indices = [i for i in range(max_len) if i not in linking]
                for longer_partition_i in range(max_len):
                    if longer_partition_i in linking:
                        shorter_partition_i = linking.index(longer_partition_i)
                        shorter_partition_seq = shorter_partition[shorter_partition_i]
                    elif include_zeros:
                        shorter_partition_i = None
                        shorter_partition_seq = "0"
                    else:
                        continue
                    longer_partition_seq = longer_partition[longer_partition_i]
                    p1_seq = longer_partition_seq if longer_partition_number == 1 else shorter_partition_seq
                    p2_seq = shorter_partition_seq if longer_partition_number == 1 else longer_partition_seq
                    # now we have e.g. p1 seq is "wa" and p2 seq is "o", want to count this in the correspondence counts
                    key = (p1_seq, p2_seq)  # the correspondence ordering should always be language 1 then language 2
                    if key not in counts:
                        counts[key] = 0
                    counts[key] += 1
    return counts


@functools.lru_cache(maxsize=10)
def get_binary_strings(n_bits):
    res = []
    n_partitions = 2 ** n_bits
    for partition_i in range(n_partitions):
        binary = format(partition_i, "b").rjust(n_bits, "0")
        res.append(binary)
    return res


def get_partitions(w):
    n = len(w)
    if n == 1:
        return [w]
    if n == 0:
        raise ValueError("empty string")

    # idea from https://stackoverflow.com/a/25460561
    n_potential_breaks = n - 1  # fencepost
    binary_strings = get_binary_strings(n_bits=n_potential_breaks)
    partitions = []
    for b in binary_strings:
        partition = convert_binary_into_partition(b, w)
        partitions.append(partition)
    return partitions


def convert_binary_into_partition(binary, s):
    # partition s into substrings based on the binary (which says, at each boundary between chars, 1 if put a divide else 0)
    assert len(binary) == len(s) - 1, f"{s}\t{binary}"
    res = []
    current_substr = ""
    for i in range(len(binary)):
        # binary bit i goes right after s char i
        bit = binary[i]
        char = s[i]
        current_substr += char
        if bit == "1":
            # put a divider
            res.append(current_substr)
            current_substr = ""
        else:
            assert bit == "0", bit
            # just add to the substring and continue
            continue
    current_substr += s[-1]
    res.append(current_substr)
    return res


def print_most_common_correspondences(correspondences, max_n=20):
    items = [(v, k) for k, v in correspondences.items()]
    items = sorted(items, reverse=True)
    for count, correspondence in items[:max_n]:
        print(f"{correspondence} : {count}")


if __name__ == "__main__":
    proto_words = get_proto_words(500)
    l1_words, l1_sound_changes = mutate_words(proto_words, random.randint(10, 40))
    l2_words, l2_sound_changes = mutate_words(proto_words, random.randint(10, 40))
    l3_words, l3_sound_changes = mutate_words(proto_words, random.randint(10, 40))
    l4_words, l4_sound_changes = mutate_words(proto_words, random.randint(10, 40))

    print(l1_sound_changes)
    print(l2_sound_changes)
    print(l3_sound_changes)
    print(l4_sound_changes)

    max_len = max(max(len(w) for w in wl) for wl in [l1_words, l2_words, l3_words, l4_words])
    j = lambda s: s.ljust(max_len, " ") + "\t"
    for wp, w1, w2, w3, w4 in zip(proto_words, l1_words, l2_words, l3_words, l4_words):
        print(f"*{j(wp)}{j(w1)}{j(w2)}{j(w3)}{j(w4)}")

    correspondences = count_correspondences(l1_words, l2_words, include_zeros=False)
    # when including zeros, most of them show up as very common but are spurious
    print_most_common_correspondences(correspondences)
