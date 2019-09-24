class Lexicon:
    def __init__(self, lexemes):
        self.lexemes = []
        self.next_lexeme_designation = 0
        for lex in lexemes:
            self.add_lexeme(lex)
    
    def add_lexeme(self, lexeme):
        self.lexemes.append(lexeme)
        self.next_lexeme_designation += 1

    def all_forms(self):
        res = []
        for lex in self.lexemes:
            res += lex.forms
        return res
