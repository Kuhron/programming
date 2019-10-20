class Phoneme:
    def __init__(self, symbol):
        self.symbol = symbol
        self.allophone_rules = []
    
    def add_allophone_rule(self, rule):
        self.allophone_rules.append(rule)
