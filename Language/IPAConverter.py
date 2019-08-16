import os
import pickle

import PhoneticFeatureSpace


def get_ipa_symbol_to_features_dict():
        # FIXME: phonemes it doesn't know about: l
        if os.path.isfile("/home/wesley/programming/IPA_SYMBOL_TO_FEATURES.pickle"):
            with open("/home/wesley/programming/IPA_SYMBOL_TO_FEATURES.pickle", "rb") as f:
                d = pickle.load(f)
            return d

        else:
            def my_product(dicts):
                # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
                return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

            d = {k: sorted(v.keys(), reverse=True) for k, v in PhoneticFeatureSpace.FEATURE_KEYS.items()}
            n = 1
            for lst in d.values():
                n *= len(lst)
            input("total symbol-dict pairs to compute: {0}\npress enter to continue".format(n))

            result = {}
            i = 0
            for features_dict in my_product(d):
                phone = Phone(features_dict)
                phone = phone.restrict_features()
                i += 1
                if i % 10000 == 0:
                    print(i)
                symbol = phone.get_ipa_symbol()
                result[symbol] = phone.features

            with open("C:/Users/Wesley/Desktop/Programming/IPA_SYMBOL_TO_FEATURES.pickle", "wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

            return result


IPA_SYMBOL_TO_FEATURES = get_ipa_symbol_to_features_dict()

def get_features_from_symbol(symbol):
    return IPA_SYMBOL_TO_FEATURES[symbol]