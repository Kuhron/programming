class OrthographyConverter:
    def __init__(self, language):
        self.grapheme_to_phoneme = OrthographyConverter.initial_dict()
        self.phoneme_to_grapheme = OrthographyConverter.initial_dict()
        self.language = language

    @staticmethod
    def initial_dict():
        # return {"": "", "#": "#"}  # causes bugs where it wants to match # before letters
        return {}

    def add_grapheme_to_classes(self, grapheme, classes):
        self.language.add_grapheme(grapheme, classes)

    def add_pair(self, grapheme_str, phoneme_str):
        assert phoneme_str not in self.phoneme_to_grapheme or self.phoneme_to_grapheme[phoneme_str] == grapheme_str, "Warning: overwriting phoneme_str {} (current ortho {}, would be replaced by {})".format(phoneme_str, self.phoneme_to_grapheme[phoneme_str], grapheme_str)
        assert grapheme_str not in self.grapheme_to_phoneme or self.grapheme_to_phoneme[grapheme_str] == phoneme_str, "Warning: overwriting grapheme_str {} (current pronunciation {}, would be replaced by {})".format(grapheme_str, self.grapheme_to_phoneme[grapheme_str], phoneme_str)
        self.grapheme_to_phoneme[grapheme_str] = phoneme_str
        self.phoneme_to_grapheme[phoneme_str] = grapheme_str

    def convert_graphemes_to_phonemes(self, grapheme_str):
        grapheme_str = "#" + grapheme_str + "#"
        res = OrthographyConverter.greedy_replace(grapheme_str, self.grapheme_to_phoneme)
        return "/" + res.replace("#", "") + "/"

    def convert_phonemes_to_graphemes(self, phoneme_str):
        phoneme_str = "#" + phoneme_str + "#"
        res = OrthographyConverter.greedy_replace(phoneme_str, self.phoneme_to_grapheme)
        return "<" + res.replace("#", "") + ">"

    @staticmethod
    def greedy_replace(s, dct, keys=None):
        if s in ["", "#"]:
            return s
        if keys == None:
            keys = sorted(dct.keys(), key=lambda x: -1*len(x))  # TODO: sort keys by ranking of some kind, where they are allowed to tie, which then creates a branch of possible readings
        for i, k in enumerate(keys):
            # print("checking if {} contains {}".format(s, k))
            pattern_found = OrthographyConverter.pattern_find(k, s)
            if pattern_found is None:
                # print("it does not")
                continue
            else:
                # print("it does!")
                pre, match, post = pattern_found
                replacement = dct[match]
                # print("s {}, k {}, {}-{}-{} --> {}-{}-{}".format(s, k, pre, match, post, pre, replacement, post))
                # input("press to continue")
                remaining_keys = keys[i:]  # don't get rid of current key in case it occurs again
                replace_pre = OrthographyConverter.greedy_replace(pre, dct, remaining_keys)
                replace_post = OrthographyConverter.greedy_replace(post, dct, remaining_keys)
                return replace_pre + replacement + replace_post
        # if we fall through to here, no matches were found
        # print("s {}, but not found".format(s))
        # input("press to continue")
        return "?"

    @staticmethod
    def pattern_find(pattern, s):
        # simple substr for now
        # TODO: use regex here, make a conversion from phonology notation in CWG to regex
        if pattern not in s:
            return None
        i = s.index(pattern)
        j = i + len(pattern)
        pre = s[:i]
        match = s[i:j]
        post = s[j:]
        return pre, match, post