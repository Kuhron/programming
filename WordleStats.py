import random
import string


class WordSearchDataStructure:
    # meant for efficiently checking whether the lines in word_frequencies_en_full are about a given set of words
    def __init__(self, words):
        allowed_chars = string.ascii_uppercase
        assert all(all(c in allowed_chars for c in w) for w in words), "disallowed chars"
        self.words = words
        self.path_dict = self.create_path_dict()

    def create_path_dict(self):
        # want to query a line by looking at first char
        # if no word in our set has that, then stop and return no match
        # if some words do, then look at second char, see if words in our set *which begin with the first letter* have that second char, etc.
        # make a tree path dictionary to accomplish this
        # end of a path is a value of empty dict
        # so word "tree" looks like: {"T": {"R": {"E": {"E": {}}}}}
        # then add word "try" to get: {"T": {"R": {"E": {"E": {}}, "Y": {}}}}
        d = {}
        for w in self.words:
            d = WordSearchDataStructure.add_word_to_path_dict(w, d)

        check1 = sorted(set(self.words))
        check2 = sorted(set(WordSearchDataStructure.get_words_from_path_dict(d)))
        # print(check1)
        # print(check2)
        assert check1 == check2
        return d

    @staticmethod
    def add_word_to_path_dict(w, d):
        w = w + "#"  # need the word boundary char to say when words end
        sub_d = d
        for c in w:
            if c not in sub_d:
                sub_d[c] = {}
            sub_d = sub_d[c]
        # hopefully this modification of the element dictionaries by reference will work
        return d

    @staticmethod
    def get_words_from_path_dict(d):
        if d == {}:
            return [""]  # base case
        words = []
        for char in d:
            sub_d = d[char]
            # print(char, sub_d)
            sub_words = WordSearchDataStructure.get_words_from_path_dict(sub_d)
            # print(f"sub_words: {sub_words}")
            # input("A")
            words_this_char = [char + w.replace("#", "") for w in sub_words]
            words += words_this_char
        return words

    def contains(self, word):
        d = self.path_dict
        for c in word:
            if c not in d:
                return False
            else:
                d = d[c]
        # now check that a path ends here and it's not just a sub-word with continuations (e.g. "continu" would be one such false positive if we don't check word ending)
        return "#" in d


def get_all_words(length):
    words = set()

    with open("cmudict.txt") as f:
        lines = f.readlines()
    words1 = [l.split()[0] for l in lines]
    allowed_chars = string.ascii_uppercase
    words1 = [w.upper() for w in words1]
    words1 = ["".join(c for c in w if c in allowed_chars) for w in words1]
    words1 = [w for w in words1 if len(w) == length]
    words |= set(words1)

    # check also other words files
    with open("word_frequencies_en_full.txt") as f:
        lines = f.readlines()
    words2 = set()
    for line in lines:
        try:
            word, freq = line.strip().split()
        except ValueError:
            continue
        word = word.upper()
        if len(word) == length and all(c in allowed_chars for c in word):
            words2.add(word)
    new_words = words2 - words
    # print(f"got {len(new_words)} new words in file 2")
    words |= words2

    assert all(len(w) == length for w in words), [w for w in words if len(w) != length]
    return words


def get_frequencies(words):
    # look at a corpus
    frequency_file = "word_frequencies_en_full.txt"
    with open(frequency_file) as f:
        lines = f.readlines()
    word_search = WordSearchDataStructure(words)
    frequencies = {}
    for line in lines:
        try:
            word, freq = line.strip().split()
        except ValueError:
            continue
        word = word.upper()
        if word_search.contains(word):
            # assert word not in frequencies, f"conflict with word {word}, old freq {frequencies[word]}, new {int(freq)}"
            if word in frequencies:
                frequencies[word] += int(freq)
            else:
                frequencies[word] = int(freq)
    return frequencies


if __name__ == "__main__":
    length = 5
    words = get_all_words(length)
    frequencies = get_frequencies(words)
    max_frequency = max(frequencies.values())
    print(frequencies)

    w199 = "siege"
    w200 = "tiger"
    w201 = "banal"
    path201w = ["shift", "candy", "rouge", "blimp", "banal"]
    path201n = ["steam", "straw", "iliad", "banal"]
    w202 = "slump"
    path202w = ["stain", "chore", "plumy", "slump"]
    path202n = ["lined", "black", "plows", "slurp", "slump"]

