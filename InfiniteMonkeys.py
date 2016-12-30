import random
import string


def get_words(min_length):
    with open("words.txt") as f:
        lines = f.readlines()
    words = [x.strip() for x in lines]
    return sorted([x.lower() for x in words if x.isalnum() and len(x) >= min_length])


def get_random_word(min_length):
    s = get_random_letter()
    while len(s) < min_length or random.random() < 0.5:
        s += get_random_letter()
    return s


def get_random_letter():
    return random.choice(string.ascii_lowercase)


class WordTree:
    def __init__(self, words):
        self.words = words
        self.tree = self.create_tree()

    def create_tree(self):
        result = {}
        finished_words = []
        for word in self.words:
            if len(word) <= 1:
                assert type(word) is str
                finished_words.append(word)
            else:
                if word[0] in result:
                    result[word[0]].append(word[1:])
                else:
                    result[word[0]] = [word[1:]]
        for letter in result:
            result[letter] = WordTree(result[letter])
        return (finished_words, result)

    def __contains__(self, other):
        finished_words, sub_tree_dict = self.tree
        if other in finished_words:
            return True
        elif len(other) == 0:
            return False
        if other[0] not in sub_tree_dict:
            return False
        return other[1:] in sub_tree_dict[other[0]]




if __name__ == "__main__":
    min_length = 7

    words = get_words(min_length)
    print("building WordTree")
    word_tree = WordTree(words)
    print("done!\n")

    while True:
        word = get_random_word(min_length)
        if word in word_tree:  # good data structure
        # if word in words:  # much slower
            print(word)
        else:
            # print("~   " + word)
            pass