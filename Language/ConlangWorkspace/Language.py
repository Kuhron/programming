class Language:
    def __init__(self, name, lexicon):
        self.name = name
        self.lexicon = lexicon
        self.phoneme_classes = {}  # to be populated by commands later
        self.update_used_phonemes()

    def update_used_phonemes(self):
        forms = self.lexicon.all_forms()
        res = set()
        for w in forms:
            res |= w.get_phonemes_used()
        self.used_phonemes = res

    def add_phoneme(self, phoneme_symbol, classes_of_this_phoneme):
        for cl in classes_of_this_phoneme:
            if cl not in self.phoneme_classes:
                self.phoneme_classes[cl] = set()
            self.phoneme_classes[cl].add(phoneme_symbol)

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
