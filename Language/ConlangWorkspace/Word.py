class Word:
    def __init__(self, lst, designation=None, gloss=None):
        self.designation = designation
        assert type(lst) is list, "Word.lst must be a list, got {}: {}".format(type(lst), lst)
        self.lst = lst
        self.gloss = gloss
        
    def designate(self, designation):
        self.designation = designation

    def has_designation(self):
        return self.designation is not None

    def get_soundgloss_identifier(self):
        res = "{}:{}".format(self.to_str(), self.gloss)
        assert " " not in res
        return res
        
    @staticmethod
    def from_str(s, designation=None):
        if type(s) is not str:
            raise TypeError("expected str, got {}".format(type(s)))
        lst =  parse_word_str_to_list(s)
        assert type(lst) is list
        return Word(lst, designation)
        
    def to_str(self):
        return "".join(self.lst)
        
    def get_phonemes_used(self):
        return set(self.lst)
        
    def with_word_boundaries(self):
        if self.has_word_boundaries():
            return self
        else:
            lst = ["#"] + self.lst + ["#"]
            return Word(lst, self.designation + "#")
            
    def without_word_boundaries(self):
        if self.has_word_boundaries():
            lst = self.lst[1:-1]
            return Word(lst, self.designation.replace("#",""))
        else:
            return self
            
    def has_word_boundaries(self):
        if "#" in self.lst:
            if self.lst.count("#") == 2 and self.lst[0] == "#" and self.lst[-1] == "#":
                assert "#" in self.designation
                return True
            else:
                raise Exception("word has invalid word boundary positions: {}".format(self))
        else:
            assert "#" not in self.designation
            return False
            
    def __repr__(self):
        # don't put with(out)_word_boundaries() in here because it will call this if it throws an error, causing stack overflow
        return "Word #{} : {}".format(self.designation, "".join(self.lst))
     
    def __len__(self):
        # will count word boundaries if they are present, so be sure to get the length you want by adding or removing boundaries first
        return len(self.lst)
         
    def __getitem__(self, index):
        return self.lst[index]
        
    def __contains__(self, item):
        return item in self.lst
        
    def __eq__(self, other):
        return self.designation == other.designation and self.to_str() == other.to_str()

    def __hash__(self):
        return hash((self.designation, self.to_str()))
