class Inventory:
    def __init__(self, phonemes):
        self.phonemes = phonemes

    @staticmethod
    def random():
        feature_space = PhoneticFeatureSpace.get_random_feature_value_sets()
        print("phonetic feature space:")
        feature_space.print()
        input()
        raw_inventory = Inventory([Phone(feature_space.get_features_from_possible_values()) for i in range(40)])
        print("raw inventory ({} phonemes):".format(len(raw_inventory.phonemes)))
        raw_inventory.print()
        input()
        seen = []
        seen_symbols = []
        for phone in raw_inventory.phonemes:
            restricted_phone = phone.restrict_features()
            symbol = restricted_phone.get_ipa_symbol()
            print("new symbol: {}\nexisting symbols: {}".format(symbol, seen_symbols))
            if symbol not in seen_symbols and "_" not in symbol and "?" not in symbol:
                seen.append(restricted_phone)
                seen_symbols.append(symbol)
        inventory = Inventory(seen)
        print("final inventory:")
        inventory.print()
        input()
        return inventory


    @staticmethod
    def from_lexicon(lexicon):
        added_phoneme_symbols = set()
        for word in lexicon.words:
            for syll in word.syllables:
                for symbol in syll.phonemes:
                    added_phoneme_symbols.add(symbol)

        phonemes = []
        for symbol in added_phoneme_symbols:
            try:
                phoneme = Phone.from_ipa_symbol(symbol)

            except KeyError:
                features_dict = {}
                print("the phoneme {} was not found in the IPA symbols. Please specify what it is:".format(symbol))
                for k, d in PhoneticFeatureSpace.FEATURE_KEYS.items():
                    print("{}: {}".format(k, d))
                    while True:
                        inp = input("choice for this feature: ")
                        try: 
                            choice = int(inp.strip())
                            if choice not in d:
                                print("that choice is not valid, must be one of {}".format(sorted(d.keys())))
                                continue
                            features_dict[k] = choice
                            break
                        except ValueError:
                            print("invalid int")
                            continue
                phoneme = Phone(features_dict)

            phonemes.append(phoneme)

        inventory = Inventory(phonemes)
        print("resulting inventory:")
        inventory.print()
        input()

        return inventory

    def print(self):
        print(self.str())
        print()

    def str(self):
        delim = " , "
        return delim.join(phone.str() for phone in self.phonemes)
