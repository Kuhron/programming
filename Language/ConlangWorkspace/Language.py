from Phone import Phone
from Phoneme import Phoneme
from Grapheme import Grapheme
from SegmentSet import SegmentSet


class Language:
    def __init__(self, name, lexicon):
        self.name = name
        self.lexicon = lexicon
        self.phones = {}
        self.phonemes = {}
        self.graphemes = {}
        # self.phoneme_classes = {}  # to be populated by commands later
        self.symbol_dict = {}
        # self.symbol_classes = {}
        self.add_universal_pseudosegments()
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
        self.symbol_dict[phone.to_str()] = phone

    def add_phoneme(self, phoneme, classes_of_this_phoneme):
        assert phoneme.symbol not in self.phonemes, "already have phoneme with symbol {}".format(phoneme.symbol)
        self.phonemes[phoneme.symbol] = phoneme
        for cl in classes_of_this_phoneme:
            assert cl[0] == cl[-1] == "/", "invalid phoneme class {}".format(cl)
            if cl not in self.symbol_dict:
                self.symbol_dict[cl] = SegmentSet(Phoneme, cl)
            self.symbol_dict[cl].add(phoneme)
        self.symbol_dict[phoneme.to_str()] = phoneme

    def add_grapheme(self, grapheme, classes_of_this_grapheme):
        for cl in classes_of_this_grapheme:
            assert cl[0] == "<" and cl[-1] == ">", "invalid grapheme class {}".format(cl)
            if cl not in self.symbol_dict:
                self.symbol_dict[cl] = SegmentSet(Grapheme, cl)
            self.symbol_dict[cl].add(grapheme)
        self.symbol_dict[grapheme.to_str()] = grapheme

    def add_universal_pseudosegments(self):
        self.add_phone(Phone(".", {"syllable_boundary": 1}))
        self.add_phoneme(Phoneme("."), [])
        self.add_grapheme(Grapheme("."), [])
        self.add_phone(Phone("#", {"word_boundary": 1}))
        self.add_phoneme(Phoneme("#"), [])
        self.add_grapheme(Grapheme("#"), [])

    @staticmethod
    def unbracket_phoneme(p):
        assert type(p) is Phoneme, "can't unbracket object of type {}: {}".format(type(p), p)
        return p.symbol.replace("[","").replace("]","")

    def get_phoneme_classes(self):
        classes = [v for v in self.symbol_dict.values() if type(v) is SegmentSet and v.element_type is Phoneme]
        return sorted(classes, key=lambda s: s.symbol)

    def get_phonemes(self):
        return set(self.phonemes.values())

    def get_used_phonemes(self):
        return self.used_phonemes
