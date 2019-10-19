class Phoneme:
    def __init__(self, symbol):
        self.symbol = symbol
    
    def add_allophone_rule(self, inp, outp):
        raise NotImplementedError
