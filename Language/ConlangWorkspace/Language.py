class Language:
    def __init__(self, name, lexicon):
        self.name = name
        self.lexicon = lexicon
        self.phones = {}
        self.phonemes = {}
        self.phoneme_classes = {}  # to be populated by commands later
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
            if cl not in self.phoneme_classes:
                self.phoneme_classes[cl] = set()
            self.phoneme_classes[cl].add(phoneme.symbol)

    @staticmethod
    def unbracket_phoneme(p):
        return p.replace("[","").replace("]","")

    def get_phoneme_classes(self):
        return sorted(self.phoneme_classes.keys(), key=Language.unbracket_phoneme)

    def get_phonemes(self):
        res = set()
        for cl in self.phoneme_classes:
            for p in self.phoneme_classes[cl]:
                res.add(p)
        return sorted(res)

    def get_used_phonemes(self):
        return self.used_phonemes
