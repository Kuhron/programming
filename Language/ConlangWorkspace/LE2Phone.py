class Phone:
    def __init__(self, features_dict):
        assert all(type(v) is int for v in features_dict.values())
        self.features = features_dict

    def print(self, verbose=False):
        print(self.str(verbose=verbose))

    def str(self, verbose=False):
        if verbose:
            s = ""
            d = self.features
            get_name = lambda k, v: PhoneticFeatureSpace.FEATURE_KEYS[k][v]
            for k, val in d.items():
                if type(val) is int:
                    translated_val = get_name(k, val)
                else:
                    raise TypeError("feature dict values were not list or int, but {}".format(type(val)))
                s += "{}: {}\n".format(k, translated_val)
            return s
        else:
            return self.get_ipa_symbol()

    def get_ipa_symbol(self):
        return Phone.get_ipa_symbol_from_features(self.features)

    @staticmethod
    def from_ipa_symbol(symbol):
        if symbol == "#":
            return WordBoundaryPhone()
        else:
            return Phone(IPAConverter.get_features_from_symbol(symbol))

    @staticmethod
    def get_ipa_symbol_from_features(features):
        if type(features) is not dict:
            raise TypeError("features must be dict if using this method; if possible, use get_ipa_symbol method of Phone object")

        if features.get("is_word_boundary") == 1:
            return "#"

        phone = Phone(features)

        phone = phone.restrict_features()

        if phone.features["syllabicity"] == 0:
            return Phone.get_ipa_consonant_symbol_from_features(phone.features)

        if phone.features["syllabicity"] == 1:
            return Phone.get_ipa_vowel_symbol_from_features(phone.features) + "\u032f"

        if phone.features["syllabicity"] == 2:
            return Phone.get_ipa_consonant_symbol_from_features(phone.features) + "\u0329"

        elif phone.features["syllabicity"] == 3:
            return Phone.get_ipa_vowel_symbol_from_features(phone.features)

        else:
            raise ValueError("invalid syllabicity of {} for features_dict\n{}".format(phone.features["syllabicity"], features))

    @staticmethod
    def get_ipa_consonant_symbol_from_features(features, with_secondaries=True):
        # code = Phone.get_numerical_code_from_features(features)
        code = "_"
        secondaries = Phone.get_secondary_symbols_from_features(features) if with_secondaries else ""

        if features["c_manner"] == 0 and features["nasalization"] == 1:
            symbol = ["m", "\u0271", "n\u032a", "n", "n\u0320", "\u0273", "\u0272", "\u014b", "\u0274", code, code, code][features["c_place"]]
            if features["voicing"] == 0:
                symbol += ["\u0325", "\u030a", "\u0325", "\u0325", "\u0325", "\u030a", "\u030a", "\u030a", "\u0325", code, code, code][features["c_place"]]
            return symbol + secondaries

        if features["c_manner"] == 0:
            if features["voicing"] == 0:
                symbol = ["p", "p\u032a", "t\u032a", "t", "t\u0320", "\u0288", "c", "k", "q", code, "\u02a1", "\u0294"][features["c_place"]]
            elif features["voicing"] == 1:
                symbol = ["b", "b\u032a", "d\u032a", "d", "d\u0320", "\u0256", "\u025f", "g", "\u0262", code, code, code][features["c_place"]]
            return symbol + secondaries

        if features["c_manner"] == 1:
            plosive_features = deepcopy(features)
            plosive_features["c_manner"] = 0
            plosive_symbol = Phone.get_ipa_consonant_symbol_from_features(plosive_features, with_secondaries=False)

            fricative_features = deepcopy(features)
            fricative_features["c_manner"] = 2
            fricative_symbol = Phone.get_ipa_consonant_symbol_from_features(fricative_features, with_secondaries=False)

            tie_bar_above = True
            tie_symbol = "\u0361" if tie_bar_above else "\u035c"
            symbol = plosive_symbol + tie_symbol + fricative_symbol

            return symbol + secondaries

        if features["c_manner"] == 2:
            if features["voicing"] == 0:
                symbol = ["\u0278", "f", "\u03b8", "s", "\u0283", "\u0282", "\u00e7", "x", "\u03c7", "\u0127", "\u029c", "h"][features["c_place"]]
            elif features["voicing"] == 1:
                symbol = ["\u03b2", "v", "\u00f0", "z", "\u0292", "\u0290", "\u029d", "\u0263", "\u0281", "\u0295", "\u02a2", "\u0266"][features["c_place"]]
            return symbol + secondaries

        if features["c_manner"] == 3:
            if features["c_labialization"] == 1 and features["c_place"] in [6, 7]:
                symbol = [None, None, None, None, None, None, "\u0265", "w", None, None, None, None][features["c_place"]]
            else:
                symbol = ["\u03b2\u031e", "\u028b", "\u0279\u032a", "\u0279", "\u0279\u0320", "\u027b", "j", "\u0270", code, code, code, code][features["c_place"]]
            if features["voicing"] == 0:
                symbol += ["\u0325", "\u0325", "\u0325", "\u0325", "\u0325", "\u030a", "\u030a", "\u030a", "", "", "", ""][features["c_place"]]
            return symbol + secondaries

        return code

    @staticmethod
    def get_ipa_vowel_symbol_from_features(features, with_secondaries=True):
        # code = Phone.get_numerical_code_from_features(features)
        code = "_"
        secondaries = Phone.get_secondary_symbols_from_features(features) if with_secondaries else ""

        if features["v_height"] == 0:
            if features["v_roundedness"] == 0:
                symbol = ["a", "a\u0308", "\u0251"][features["v_backness"]]
            elif features["v_roundedness"] == 1:
                symbol = ["\u0276", "\u0276\u0308", "\u0252"][features["v_backness"]]
            return symbol + secondaries

        if features["v_height"] == 1:
            if features["v_roundedness"] == 0:
                symbol = ["\u00e6", "\u0250", code][features["v_backness"]]
            elif features["v_roundedness"] == 1:
                symbol = [code, code, code][features["v_backness"]]
            return symbol + secondaries

        if features["v_height"] == 2:
            if features["v_roundedness"] == 0:
                symbol = ["\u025b", "\u025c", "\u028c"][features["v_backness"]]
            elif features["v_roundedness"] == 1:
                symbol = ["\u0153", "\u025e", "\u0254"][features["v_backness"]]
            return symbol + secondaries

        if features["v_height"] == 3:
            if features["v_roundedness"] == 0:
                symbol = ["e\u031e", "\u0259", "\u0264\u031e"][features["v_backness"]]
            elif features["v_roundedness"] == 1:
                symbol = ["\u00f8\u031e", "\u0275\u031e", "o\u031e"][features["v_backness"]]
            return symbol + secondaries

        if features["v_height"] == 4:
            if features["v_roundedness"] == 0:
                symbol = ["e", "\u0258", "\u0264"][features["v_backness"]]
            elif features["v_roundedness"] == 1:
                symbol = ["\u00f8", "\u0275", "o"][features["v_backness"]]
            return symbol + secondaries

        if features["v_height"] == 5:
            if features["v_roundedness"] == 0:
                symbol = ["\u026a", code, "\u026a\u0320"][features["v_backness"]]
            elif features["v_roundedness"] == 1:
                symbol = ["\u028f", code, "\u028a"][features["v_backness"]]
            return symbol + secondaries

        if features["v_height"] == 6:
            if features["v_roundedness"] == 0:
                symbol = ["i", "\u0268", "\u026f"][features["v_backness"]]
            elif features["v_roundedness"] == 1:
                symbol = ["y", "\u0289", "u"][features["v_backness"]]
            return symbol + secondaries

        return code

    @staticmethod
    def get_secondary_symbols_from_features(features):
        result = ""

        if features["nasalization"] == 1:
            if features["c_manner"] != 0:
                result += "\u0303"

        if features["c_palatization"] == 1:
            result += "\u02b2"

        if features["c_labialization"] == 1:
            if features["c_manner"] != 3 and features["c_place"] not in [6, 7]:
                result += "\u02b7"

        if features["c_velarization"] == 1:
            result += "\u02e0"

        if features["c_pharyngealization"] == 1:
            result += "\u02e4"

        if features["c_glottalization"] == 1:
            if features["syllabicity"] in [1, 3] or features["voicing"] == 1:
                result += "\u0330"
            else:
                result += "\u02bc"

        if features["c_aspiration"] == 1:
            if features["voicing"] == 0:
                result += "\u02b0"
            else:
                result += "\u02b1"

        if features["length"] == 1:
            result += "\u02d0"

        return result

    @staticmethod
    def get_numerical_code_from_features(features):
        # return "_"
        # return repr(features)
        return "\n\n<?" + ",".join([str(v) for k, v in sorted(features.items())]) + ">\n\n"

    @staticmethod
    def get_random_features():
        d = {}
        for k, v in PhoneticFeatureSpace.FEATURE_KEYS.items():
            d[k] = random.choice([x for x in v.keys()])
        return d

    def restrict_features(self):
        features = deepcopy(self.features)

        # no nasalization while voiceless
        if features["nasalization"] == 1 and features["voicing"] == 0:
            features["nasalization"] = 0

        # uvular/pharyngeal/epiglottal approximant -> fricative
        if features["c_place"] in [8, 9, 10] and features["c_manner"] == 3:
            features["c_manner"] = 2

        # glottal approximant -> fricative
        if features["c_place"] == 11 and features["c_manner"] == 3:
            features["c_manner"] = 2

        # pharyngeal stop -> epiglottal
        if features["c_place"] == 9 and features["c_manner"] == 0:
            features["c_place"] = 10

        # voiced epiglottal/glottal stop/affricate -> voiceless
        if features["c_place"] in [10, 11] and features["c_manner"] in [0, 1] and features["voicing"] == 1:
            features["voicing"] = 0

        # labialized bilabial/labiodental -> non-labialized
        if features["c_place"] in [0, 1] and features["c_labialization"] == 1:
            features["c_labialization"] = 0

        # pharyngealized/glottalized pharyngeal/epiglottal/glottal -> non-*
        if features["c_place"] in [9, 10, 11]:
            features["c_pharyngealization"] = 0
            features["c_glottalization"] = 0

        # palatized palatal -> non-palatized
        if features["c_place"] == 6 and features["c_palatization"] == 1:
            features["c_palatization"] = 0

        # velarized velar+ -> non-velarized
        if features["c_place"] in [7, 8, 9, 10, 11] and features["c_velarization"] == 1:
            features["c_velarization"] = 0

        # aspirated h -> non-aspirated
        if features["c_place"] in [9, 10, 11] and features["c_manner"] == 2 and features["c_aspiration"] == 1:
            features["c_aspiration"] = 0

        # velarized vowels -> back
        if features["syllabicity"] in [1, 3] and features["c_velarization"] == 1:
            features["v_backness"] = 2
            features["c_velarization"] = 0

        # palatized vowels -> front
        if features["syllabicity"] in [1, 3] and features["c_palatization"] == 1:
            features["v_backness"] = 0
            features["c_palatization"] = 0

        # labialized vowels -> rounded
        if features["syllabicity"] in [1, 3] and features["c_labialization"] == 1:
            features["v_roundedness"] = 1
            features["c_labialization"] = 0

        # aspirated/pharyngealized vowel -> non-*
        if features["syllabicity"] in [1, 3]:
            features["c_aspiration"] = 0
            features["c_pharyngealization"] = 0

        # aspirated and glottalized -> glottalized
        if features["c_glottalization"] == 1 and features["c_aspiration"] == 1:
            features["c_aspiration"] = 0

        # syllabic consonants must not contain plosive
        if features["syllabicity"] == 2 and features["c_manner"] in [0, 1]:
            features["syllabicity"] = 0

        # high semivowels -> consonants, excluding central
        if features["syllabicity"] == 1 and features["v_height"] == 6 and features["v_backness"] in [0, 2]:
            features["syllabicity"] = 0
            features["c_labialization"] = features["v_roundedness"]
            if features["v_backness"] == 0:
                features["c_place"] = 6
            elif features["v_backness"] == 2:
                features["c_place"] = 7

        # aspirated approximant -> non-aspirated
        if features["c_manner"] == 3:
            features["c_aspiration"] = 0

        return Phone(features)


class WordBoundaryPhone(Phone):
    def __init__(self):
        self.features = {"is_word_boundary": 1}

    def __eq__(self, other):
        return type(other) is WordBoundaryPhone and self.features["is_word_boundary"] == other.features["is_word_boundary"] == 1

    def restrict_features(self):
        return WordBoundaryPhone()
