# generalized class for strings of things
# e.g. graphemes, phonemes, phones
# can have mixed types
# this will help with things like:
#   orthographic rules <VgV> <V/z/V>
#   allophonic rules /Vz[HV]/ /V{r}[HV]/


class SegmentSequence(list):
    @staticmethod
    def from_str(s):
        types = ["Grapheme", "Phoneme", "Phone"]
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
            if char == "/" and type_stack[-1] == "Phoneme":
                # this one acts different because the beginning and ending symbols are the same
                assert not inside_brackets
                closing_type = types[ending_symbols.index(char)]
                assert closing_type == type_stack[-1], "mismatched bracketing: {}".format(s)
                type_stack = type_stack[:-1]
            elif char in beginning_symbols:
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
                beginning_symbol = beginning_symbols[types.index(current_type)]
                ending_symbol = ending_symbols[types.index(current_type)]
                segment_str = beginning_symbol + segment_symbol + ending_symbol
                segments.append(segment_str)
            elif inside_brackets:
                current_symbol += char
            else:
                current_type = type_stack[-1]
                segment_symbol = char
                beginning_symbol = beginning_symbols[types.index(current_type)]
                ending_symbol = ending_symbols[types.index(current_type)]
                segment_str = beginning_symbol + segment_symbol + ending_symbol
                segments.append(segment_str)

        assert len(type_stack) == 1, "mismatched bracketing: {}".format(s)

        print("made segseq from str {} -> {}".format(s, segments))
        return SegmentSequence(segments)
