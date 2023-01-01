# for the product-oriented schema extractor
# make a simple semantic space (let's focus on verbs)

# bits:

# verb lexemes (hard to find a 2^n grid of semantic features for this, but just try something)
# in future implementations, could just do some one-hot encoded sets of more than two variables / lexemes

# person-marking grid
# 1>2 1>3 1>self 1>indef
# 2>1 2>3 2>self 2>indef
# 3>1 3>2 3>self 3>indef
# indef>1 indef>2 indef>3 indef>indef
# so the bits for this are:
# 1 or 2 subject / 3 or indef subject
# upper (1 or 3) subject / lower (2 or indef) subject
# 123 object / self or indef object
# upper (lower-numbered = higher-ranked of the two options) / lower object

# singular subject / plural subject
# singular object / plural object

# tense
# remote / recent
# past / nonpast
# (the resulting tenses are remote past, recent past, present, future)
# can also have aspect of continuous/momentane


import random
import numpy as np


class VerbLexemeSubVector:
    def __init__(self, bits):
        assert type(bits) is np.ndarray, type(bits)
        assert bits.shape == (4,), bits.shape
        self.bits = bits
        self.bit_string = "".join(self.bits.astype(str))
        self.gloss = VerbLexemeSubVector.get_gloss_from_bits(self.bits)

    @staticmethod
    def get_gloss_from_bits(bits):
        # for now don't worry about arranging them by semantic features, just put 16 transitive verbs
        arr = [
            "see", "talk.to", "eat", "look.for",
            "hear", "think.about", "feed", "chase",
            "touch", "meet", "deceive", "follow",
            "love", "recognize", "stop", "hit",
        ]
        powers = 2 ** np.array([3, 2, 1, 0])
        index = (bits * powers).sum()
        return arr[index]

    @staticmethod
    def random():
        bits = np.random.choice([0, 1], (4,))
        return VerbLexemeSubVector(bits)


class PersonMarkingSubVector:
    def __init__(self, bits):
        assert type(bits) is np.ndarray, type(bits)
        assert bits.shape == (6,), bits.shape
        self.bits = bits
        self.bit_string = "".join(self.bits.astype(str))
        self.gloss = PersonMarkingSubVector.get_gloss_from_bits(self.bits)

    @staticmethod
    def get_gloss_from_bits(bits):
        b_12_or_34_subj, b_upper_or_lower_subj, b_123_or_r4_obj, b_upper_or_lower_obj, b_subj_num, b_obj_num = bits
        
        subj_person_str = [["1", "2"], ["3", "indef"]][b_12_or_34_subj][b_upper_or_lower_subj]
        subj_num_str = ["sg", "pl"][b_subj_num]
        subj_str = subj_person_str + "." + subj_num_str

        obj_person_str_grid = [
            [
                [["2", "3"], ["1", "indef"]],
                [["1", "3"], ["2", "indef"]],
            ], [
                [["1", "2"], ["3", "indef"]],
                [["1", "2"], ["3", "indef"]],
            ]
        ]
        obj_person_str = obj_person_str_grid[b_12_or_34_subj][b_upper_or_lower_subj][b_123_or_r4_obj][b_upper_or_lower_obj]
        obj_num_str = ["sg", "pl"][b_obj_num]
        obj_str = obj_person_str + "." + obj_num_str

        return f"{subj_str}>{obj_str}"

    @staticmethod
    def random():
        bits = np.random.choice([0, 1], (6,))
        return PersonMarkingSubVector(bits)


class TenseAspectMarkingSubVector:
    def __init__(self, bits):
        assert type(bits) is np.ndarray, type(bits)
        assert bits.shape == (3,), bits.shape
        self.bits = bits
        self.bit_string = "".join(self.bits.astype(str))
        self.gloss = TenseAspectMarkingSubVector.get_gloss_from_bits(self.bits)

    @staticmethod
    def get_gloss_from_bits(bits):
        b_past_nonpast, b_remote_recent, b_cont_mom = bits
        tense_str = [["rem.pst", "rec.pst"], ["fut", "prs"]][b_past_nonpast][b_remote_recent]
        aspect_str = ["cont", "moment"][b_cont_mom]
        return tense_str + "." + aspect_str

    @staticmethod
    def random():
        bits = np.random.choice([0, 1], (3,))
        return TenseAspectMarkingSubVector(bits)


class MeaningVector:
    def __init__(self, bits):
        assert type(bits) is np.ndarray, type(bits)
        assert bits.shape == (13,), bits.shape
        self.bits = bits
        self.bit_string = "".join(self.bits.astype(str))
        self.verb_lexeme_subvector = VerbLexemeSubVector(self.bits[0:4])
        self.person_marking_subvector = PersonMarkingSubVector(self.bits[4:10])
        self.tense_aspect_marking_subvector = TenseAspectMarkingSubVector(self.bits[10:13])
        self.gloss = "{lex}-{p}-{tam}".format(
            lex = self.verb_lexeme_subvector.gloss,
            p = self.person_marking_subvector.gloss,
            tam = self.tense_aspect_marking_subvector.gloss,
        )

    @staticmethod
    def random():
        bits = np.random.choice([0, 1], (13,))
        return MeaningVector(bits)


if __name__ == "__main__":
    v = MeaningVector.random()
    print(v.bit_string)
    print(v.gloss)
