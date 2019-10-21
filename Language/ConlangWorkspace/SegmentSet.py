from Phone import Phone
from Phoneme import Phoneme
from Grapheme import Grapheme


class SegmentSet(set):
    def __init__(self, element_type, symbol):
        super().__init__()
        self.element_type = element_type
        self.symbol = symbol

    def add(self, item):
        assert type(item) is self.element_type, "can't add item of type {} to SegmentSet of {}".format(type(item), self.element_type)
        super().add(item)

    def __repr__(self):
        symbols = [x.symbol for x in self]
        if self.element_type is Phone:
            a, b = "\{", "\}"
        elif self.element_type is Phoneme:
            a, b = "/", "/"
        elif self.element_type is Grapheme:
            a, b = "<", ">"
        else:
            print("Warning: SegmentSet of another type: {}".format(self.element_type))
            a, b = "\{\{", "\}\}"
        return a + ", ".join(sorted(symbols)) + b