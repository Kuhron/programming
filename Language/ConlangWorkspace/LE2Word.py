class LE2Word:
    def __init__(self, syllables):
        assert type(syllables) is list and all(type(x) is Syllable for x in syllables), "invalid syllables passed to Word: {}".format(syllables)
        self.syllables = syllables

    def __add__(self, other):
        if type(other) is LE2Word:
            return LE2Word(self.syllables + other.syllables)  # TODO: re-syllabify if necessary
        else:
            return NotImplemented

    def get_phoneme_list(self):
        lst = []
        for syll in self.syllables:
            lst.extend(syll.phonemes)
        return lst

    def get_ipa_str(self):
        phoneme_list = self.get_phoneme_list()
        return "".join(phone.get_ipa_symbol() for phone in phoneme_list)

    @staticmethod
    def from_user_input():
        print("Input a word to add to the language.\nFormat: phonemes separated by spaces, syllables separated by hyphens or dollar signs")
        print("Example: k a - t i")
        inp = input("word: ")
        inp = inp.replace("$", "-").strip()
        if inp == "":
            return []
        syllables = inp.split("-")
        syllables = [syll.strip().split() for syll in syllables]
        syllables = [Syllable(phones_lst) for phones_lst in syllables]
        return LE2Word(syllables)

    @staticmethod
    def from_string(s):
        word = []
        for symbol in s:
            if symbol not in ["-", "\ufeff"]:
                phone = Phone.from_ipa_symbol(symbol)
                word.append(phone)
        return word

    @staticmethod
    def from_phone_list(lst):
        # TODO: add ability to syllabify the list given a Phonology object
        syllables = [Syllable(lst)]
        return LE2Word(syllables)

    @staticmethod
    def get_random_phone_sequence(n_syllables, inventory, syllable_structure):
        syllables = []
        for i in range(n_syllables):
            syllable_phonemes = []
            for typ in syllable_structure:
                if typ == "C":
                    cands = [x for x in inventory.phonemes if x.features["syllabicity"] in [0, 1]]
                    chosen = random.choice(cands) if cands != [] else None
                elif typ == "V":
                    cands = [x for x in inventory.phonemes if x.features["syllabicity"] in [2, 3]]
                    chosen = random.choice(cands) if cands != [] else None
                elif typ == "N":
                    cands = [x for x in inventory.phonemes if x.features["syllabicity"] == 0 and x.features["nasalization"] == 1 and x.features["c_manner"] == 0]
                    chosen = random.choice(cands) if cands != [] else None
                else:
                    chosen = random.choice(inventory.phonemes)
    
                if chosen is not None:
                    syllable_phonemes.append(chosen)
            syllables.append(Syllable(syllable_phonemes))
        return LE2Word(syllables)

    def print(self, as_word=False, verbose=False):
        if verbose:
            for d in lst:
                phone = Phone(d)
                print("symbol: {}".format(phone.get_ipa_symbol()))
                phone.print()
                input()
        else:
            s = ""
            for syll in self.syllables:
                for phone in syll.phonemes:
                    s += phone.get_ipa_symbol()
            print(s)
            # delim = ""
            # print(delim.join(Phone.get_ipa_symbol_from_features(d) for d in lst))
        print()
