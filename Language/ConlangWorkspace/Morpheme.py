class Morpheme:
    # just a single morpheme, such as tense
    # cannot be applied to inflect a citation form by itself, since only has one feature
    def __init__(self, pos, feature, string, gloss):
        self.pos = pos
        self.feature = feature
        self.string = string
        self.gloss = gloss
