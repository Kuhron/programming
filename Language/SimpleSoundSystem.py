# for product-oriented schema extraction, want a basic sound system to operate within
# would love to do articulators and stuff, and give it real phonetics that will affect the words via change/epenthesis/etc.,
# but for now start with something very simple so can focus on the schema extraction part

# each vector is a CV syllable, each feature binary
#
# consonants:
# peripheral vs coronal
# back vs front
# stop vs continuant
# oral/fricative-like vs nasal/liquid-like
# 
# so the grid of consonants is (where position number is the encoding in these four features):
#  k ŋg  h  ŋ
#  p mb  w  m
#  c ɲɟ  j  ɲ
#  t nd  s  n
# NOTE: I may change this because I don't like how there are too many nasals and no liquids
#
# vowels:
# front/back
# high/low
# 
# so the grid of vowels is:
# i e
# u a
#
# the whole vector for a syllable has 6 bits

# each word is a sequence of some number of these syllables (might use RNN for pattern extraction? not sure yet)


import random
import numpy as np


class SoundVector:
    def __init__(self, bits):
        assert type(bits) is np.ndarray, type(bits)
        assert bits.shape == (6,), bits.shape
        assert np.isin(bits, [0, 1]).all()
        self.bits = bits
        self.bit_string = "".join(self.bits.astype(str))
        self.symbols = SoundVector.get_symbols_from_bits(self.bits)
        self.string = "".join(self.symbols)

    @staticmethod
    def get_symbols_from_bits(bits):
        c_bits = bits[:4]
        v_bits = bits[4:6]
        c_symbols = [
            "k",  "ŋg",  "h", "ŋ",
            "p",  "mb",  "v", "m",
            "tʃ", "ndʒ", "r", "j",
            "t",  "nd",  "s", "n",
        ]
        v_symbols = [
            "i", "e",
            "u", "a",
        ]
        c_2powers = 2 ** np.array([3, 2, 1, 0])
        c_index = (c_bits * c_2powers).sum()
        c_str = c_symbols[c_index]
        v_2powers = 2 ** np.array([1, 0])
        v_index = (v_bits * v_2powers).sum()
        v_str = v_symbols[v_index]
        return [c_str, v_str]

    @staticmethod
    def random():
        bits = np.random.choice([0,1], (6,))
        return SoundVector(bits)

    def get_mutated(self):
        # change one bit
        bit_index = random.randrange(6)
        bits = np.copy(self.bits)  # slicing like self.bits[:] retains reference to object and allows mutation, which we don't want
        bits[bit_index] = 1 - bits[bit_index]
        assert (bits != self.bits).sum() == 1, "object was mutated"
        return SoundVector(bits)

    def __repr__(self):
        return f"<{self.bit_string} = {self.string}>"


class SoundVectorSeries:
    def __init__(self, bits):
        assert type(bits) is np.ndarray, type(bits)
        assert bits.ndim == 2, bits.shape
        assert bits.shape[-1] == 6, bits.shape
        assert np.isin(bits, [0, 1]).all()
        self.bits = bits

        self.vectors = []
        for i in range(bits.shape[0]):
            v = SoundVector(bits[i,:])
            self.vectors.append(v)

        self.bit_string = ",".join(v.bit_string for v in self.vectors)
        self.string = "".join(v.string for v in self.vectors)

    @staticmethod
    def random():
        length = np.random.randint(2, 5)
        bits = np.random.choice([0, 1], (length, 6))
        return SoundVectorSeries(bits)

    def get_mutated(self):
        vector_index = random.randrange(self.bits.shape[0])  # which vector to mutate
        bit_index = random.randrange(6)
        bits = np.copy(self.bits)
        bits[vector_index, bit_index] = 1 - bits[vector_index, bit_index]
        assert (bits != self.bits).sum() == 1, "object was mutated"
        return SoundVectorSeries(bits)

    def __repr__(self):
        return f"<{self.bit_string} = {self.string}>"


class FiveSyllableSoundVector:
    # for machine learning, pad all words, force them to be max of five syllables
    # for each syllable, there is an extra bit that says whether the syllable is actually used
    # need to adjust any random inputs such that the first is-used bit is always 1,
    # - any is-used bits after the first 0 are also 0
    # - all other bits in an unused syllable are 0
    # this will force a given surface form to only have a unique bit form, which will be good for the machine learning so it's not finding stuff in the unused bits that shouldn't matter

    def __init__(self, bits):
        assert type(bits) is np.ndarray, type(bits)
        assert bits.shape == (5 * 7,), bits.shape
        bits = FiveSyllableSoundVector.cleanup_bits(bits)  # validate and change any aberrant bits
        self.bits = bits
        self.bit_string = "".join(self.bits.astype(str))
        self.syllable_sound_vectors = [SoundVector(self.bits[i*7 + 1 : (i+1)*7]) if self.bits[i*7] else None for i in range(5)]
        self.string = "".join(syll.string if syll is not None else "X" for syll in self.syllable_sound_vectors)

    @staticmethod
    def random():
        bits = np.random.choice([0, 1], (35,))
        # do the validation in the initialization
        return FiveSyllableSoundVector(bits)

    @staticmethod
    def cleanup_bits(bits):
        new_bits = [b for b in bits]
        new_bits[0] = 1  # force the first is-on bit to be true so word is non-empty

        # find the first is-not-used and zero out everything from then on
        n_syllables = None
        for i in range(1, 5):
            is_on = bits[i*7]
            if not is_on:
                n_syllables = i  # this will be the number of syllables seen before, e.g. i=2 is the third iteration so we've already seen two before this
                break
        if n_syllables is None:
            # never got an is-not-used, so all syllables are on
            n_syllables = 5
        assert 1 <= n_syllables <= 5

        # now zero out everything starting with the first is-not-used
        is_not_used_index = 7 * n_syllables
        for i in range(is_not_used_index, len(bits)):
            new_bits[i] = 0

        return np.array(new_bits)


if __name__ == "__main__":
    # singulars = [SoundVectorSeries.random() for i in range(100)]
    # plurals = [SoundVectorSeries.random() for i in range(100)]
    # for s, p in zip(singulars, plurals):
    #     print(f"the plural of {s.string} is {p.string}")

    for i in range(100):
        v = FiveSyllableSoundVector.random()
        print(f"{v.bit_string} = {v.string}")
