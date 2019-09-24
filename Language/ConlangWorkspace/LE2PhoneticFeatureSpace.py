import random

class PhoneticFeatureSpace:
    # for when features take on multiple values in the form of lists,
    # e.g. when picking which features are contrastive in a language, before picking any phones/phonemes

    FEATURE_KEYS = {
        "c_aspiration": {0: "non-aspirated", 1: "aspirated"},
        "c_glottalization": {0: "non-glottalized", 1: "glottalized"},
        "c_labialization": {0: "non-labialized", 1: "labialized"},
        "c_lateralization": {0: "non-lateral", 1: "lateral"},
        "c_manner": {0: "stop", 1: "affricate", 2: "fricative", 3: "approximant"},
        "c_palatization": {0: "non-palatized", 1: "palatized"},
        "c_pharyngealization": {0: "non-pharyngealized", 1: "pharyngealized"},
        "c_place": {0: "bilabial", 1: "labiodental", 2: "dental", 3: "alveolar", 4: "postalveolar", 5: "retroflex", 6: "palatal", 7: "velar", 8: "uvular", 9: "pharyngeal", 10: "epiglottal", 11: "glottal"},
        "c_velarization": {0: "non-velarized", 1: "velarized"},
        "length": {0: "normal", 1: "long"},
        "nasalization": {0: "non-nasalized", 1: "nasalized"},
        "syllabicity": {0: "consonant", 1: "non-syllabic vowel", 2: "syllabic consonant", 3: "vowel"},
        "v_backness": {0: "front", 1: "central", 2: "back"},
        "v_height": {0: "open", 1: "near-open", 2: "open-mid", 3: "mid", 4: "close-mid", 5: "near-close", 6: "close"},
        "v_roundedness": {0: "unrounded", 1: "rounded"},
        "voicing": {0: "voiceless", 1: "voiced"},
        "is_word_boundary": {0: False, 1: True},
    }

    DEFAULT_FEATURE_VALUES = {k: 0 for k in FEATURE_KEYS.keys()}

    def __init__(self, features_dict):
        assert all(type(v) is list for v in features_dict.values())
        self.features = features_dict

    def get_features_from_possible_values(self):
        d = {}
        for k, v in PhoneticFeatureSpace.FEATURE_KEYS.items():
            possibilities = self.features.get(k)
            if possibilities is not None:
                d[k] = random.choice(possibilities)
        return d

    @staticmethod
    def get_random_feature_value_sets():
        value_probabilities = {
            "c_aspiration": [1, 0.3],
            "c_glottalization": [1, 0],
            "c_labialization": [1, 0],
            "c_lateralization": [1, 0.5],
            "c_manner": [1, 0.5, 0.7, 0.7],
            "c_palatization": [1, 0.3],
            "c_pharyngealization": [1, 0.1],
            "c_place": [0.5, 0.1, 0.1, 0.8, 0.3, 0.4, 0.4, 1, 0.3, 0.1, 0.05, 0.5],
            "c_velarization": [1, 0.2],
            "length": [1, 0.2],
            "nasalization": [1, 0.2],
            "syllabicity": [1, 0.05, 0.2, 1],
            "v_backness": [1, 0.5, 0.9],
            "v_height": [1, 0.2, 0.5, 0.5, 0.5, 0.2, 0.9],
            "v_roundedness": [1, 0.8],
            "voicing": [1, 0.5],
        }

        d = {}
        for k, v in PhoneticFeatureSpace.FEATURE_KEYS.items():
            if k not in value_probabilities:
                # e.g. is_word_boundary, which we don't want to be involved in phonology creation
                continue
            d[k] = []
            for val in v.keys():
                if random.random() < value_probabilities[k][val]:
                    d[k].append(val)

        return PhoneticFeatureSpace(d)

    def print(self):
        d = self.features
        print("showing feature values")
        get_name = lambda k, v: PhoneticFeatureSpace.FEATURE_KEYS[k][v]
        for k, val in d.items():
            if type(val) is list:
                translated_val = [get_name(k, v) for v in val]
            else:
                raise TypeError("feature dict values were not list or int, but {}".format(type(val)))
            print("{}: {}".format(k, translated_val))
        print()
