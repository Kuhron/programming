class Language:
    def __init__(self, name, lexicon):
        self.name = name
        self.lexicon = lexicon
        self.phones = {}
        self.phonemes = {}
        # self.phoneme_classes = {}  # to be populated by commands later
        self.symbol_classes = {}
        self.update_used_phonemes()

    def update_used_phonemes(self):
        forms = self.lexicon.all_forms()
        res = set()
        for w in forms:
            res |= w.get_phonemes_used()
        self.used_phonemes = res

    def add_phone(self, phone):
        assert phone.symbol not in self.phones, "already have phone with symbol {}".format(phone.symbol)
        self.phones[phone.symbol] = phone

    def add_phoneme(self, phoneme, classes_of_this_phoneme):
        assert phoneme.symbol not in self.phonemes, "already have phoneme with symbol {}".format(phoneme.symbol)
        self.phonemes[phoneme.symbol] = phoneme
        for cl in classes_of_this_phoneme:
            assert cl[0] == cl[-1] == "/", "invalid phoneme class {}".format(cl)
            if cl not in self.symbol_classes:
                self.symbol_classes[cl] = set()
            self.symbol_classes[cl].add(phoneme.symbol)

    @staticmethod
    def unbracket_phoneme(p):
        return p.replace("[","").replace("]","")

    def get_phoneme_classes(self):
        keys = [k for k in self.symbol_classes.keys() if k[0] == k[-1] == "/"]
        return sorted(keys, key=Language.unbracket_phoneme)

    def get_phonemes(self):
        res = set()
        for cl in self.symbol_classes:
            if cl[0] == cl[-1] == "/":
                for p in self.symbol_classes[cl]:
                    res.add(p)
        return sorted(res)

    def get_used_phonemes(self):
        return self.used_phonemes
