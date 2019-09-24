class LE2Lexicon:
    def __init__(self, words):
        self.words = words

    @staticmethod
    def from_user_input():
        print("Add words. When finished, press enter without entering anything.")
        words = []
        while True:
            w = LE2Word.from_user_input()
            if w == []:
                break
            words.append(w)
        return LE2Lexicon(words)

    @staticmethod
    def from_phonology(phonology):
        vocabulary = []
        for i in range(3):
            paradigm = get_random_paradigm(phonology.inventory, phonology.syllable_structure_set)
            for word in paradigm:
                vocabulary.append(word)
        return vocabulary
