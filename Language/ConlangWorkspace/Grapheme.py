class Grapheme:
    def __init__(self, symbol):
        assert "<" not in symbol and ">" not in symbol
        self.symbol = symbol

    def to_str(self):
        return "<" + self.symbol + ">"

    @staticmethod
    def from_str(s):
        assert s[0] == "<" and s[-1] == ">"
        return Grapheme(s[1:-1])

    def __repr__(self):
        return self.to_str()