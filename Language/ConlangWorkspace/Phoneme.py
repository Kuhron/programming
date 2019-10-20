class Phoneme:
    def __init__(self, symbol):
        assert "/" not in symbol
        self.symbol = symbol
        self.allophone_rules = []
    
    def add_allophone_rule(self, rule):
        self.allophone_rules.append(rule)

    def to_str(self):
        return "/" + self.symbol + "/"

    @staticmethod
    def from_str(s):
        assert s[0] == s[-1] == "/"
        return Phoneme(s[1:-1])

    def __repr__(self):
        return self.to_str()