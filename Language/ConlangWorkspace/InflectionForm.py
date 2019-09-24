class InflectionForm:
    # multiple morphemes, ready to be applied to a citation form and then the surface form will be finished
    def __init__(self, pos, string, gloss, designation_suffix):
        self.pos = pos
        self.string = string
        self.gloss = gloss
        self.designation_suffix = designation_suffix

    def __repr__(self):
        return "InflectionForm suffix #{} of POS {}: {} = '{}'".format(
            self.designation_suffix, self.pos, self.string, self.gloss, 
        )
