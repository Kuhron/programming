# generalized class for strings of things
# e.g. graphemes, phonemes, phones
# can have mixed types
# this will help with things like:
#   orthographic rules <VgV> <V/z/V>
#   allophonic rules /Vz[HV]/ /V{r}[HV]/

from Grapheme import Grapheme
from Phoneme import Phoneme
from Phone import Phone


class SegmentSequence:
    def __init__(self, segments):
        self.segments = segments

    @staticmethod
    def from_string(s):
        types = [Grapheme, Phoneme, Phone]
        beginning_symbols = ["<", "/", "{"]
        ending_symbols = [">", "/", "}"]
        first_symbol = s[0]
        last_symbol = s[-1]
        assert first_symbol in beginning_symbols, "cannot parse segment string with undefined primary type: {}".format(s)
        assert last_symbol == ending_symbols[beginning_symbols.index(first_symbol)], "cannot parse segment string with mismatched bracketing: {}".format(s)

        segments = []
        primary_type = types[beginning_symbols.index(first_symbol)]
        type_stack = [primary_type]
        inside_brackets = False
        current_symbol = ""
        for char in s[1:-1]:
            if char in beginning_symbols:
                assert not inside_brackets
                new_type = types[beginning_symbols.index(char)]
                type_stack.append(new_type)
            elif char in ending_symbols:
                assert not inside_brackets
                closing_type = types[ending_symbols.index(char)]
                assert closing_type == type_stack[-1], "mismatched bracketing: {}".format(s)
                type_stack = type_stack[:-1]
            elif char == "[":
                assert not inside_brackets
                inside_brackets = True
            elif char == "]":
                assert inside_brackets
                inside_brackets = False
                current_type = type_stack[-1]
                segment_symbol = current_symbol
                segment = current_type(segment_symbol)
                segments.append(segment)
            elif inside_brackets:
                current_symbol += char
            else:
                current_type = type_stack[-1]
                segment_symbol = char
                segment = current_type(segment_symbol)
                segments.append(segment)

        assert len(type_stack) == 1, "mismatched bracketing: {}".format(s)

        return SegmentSequence(segments)