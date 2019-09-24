class SoundChange:
    COMMON_SOUND_CHANGES = [
        ({"voicing": 0}, {"voicing": 1}, [{"syllabicity": 3}], [{"syllabicity": 3}]),  # intervocalic voicing
        ({"syllabicity": 3}, {"v_backness": 2}, [{"syllabicity": [0, 2], "c_place": [8, 9, 10, 11]}], []),  # vowels move back after uvulars+
    ]

    def __init__(self, from_features, to_features, phonetic_environment):
        self.from_features = from_features
        self.to_features = to_features
        self.phonetic_environment = phonetic_environment

    def apply_to_word(self, word):
        if self.from_features == "" and self.to_features == "":
            raise Exception("bad practice! features should be feature dicts, and if they do nothing, then make them dicts that somehow say that")  # FIXME
            return word

        phoneme_list = [WordBoundaryPhone()] + word.get_phoneme_list() + [WordBoundaryPhone()]

        before_features_list = self.phonetic_environment.before_environment
        after_features_list = self.phonetic_environment.after_environment
        before_len = len(before_features_list)
        start_index = before_len
        after_len = len(after_features_list)
        end_index_exclusive = len(phoneme_list) - after_len

        new_phoneme_list = []

        if type(self.from_features) is dict:
            for i, phone in enumerate(phoneme_list):
                if i < before_len or i >= end_index_exclusive:
                    new_phoneme_list.append(deepcopy(phone))
                    continue

                before_environment = phoneme_list[i - before_len: i]
                after_environment = phoneme_list[i + 1: i + after_len + 1]

                matches_phone = matches_features_dict(phone, self.from_features)
                if not matches_phone:
                    new_phoneme_list.append(deepcopy(phone))
                    continue

                matches_before = all([matches_features_dict(before_environment[j], before_features_list[j]) for j in range(before_len)])
                matches_after = all([matches_features_dict(after_environment[j], after_features_list[j]) for j in range(after_len)])
                if matches_before and matches_after:
                    if self.to_features == "":
                        new_phoneme_list.append(None)
                    else:
                        new_phone = deepcopy(phone)
                        new_phone.update(self.to_features)
                        new_phoneme_list.append(new_phone)
                else:
                    new_phoneme_list.append(deepcopy(phone))

        elif self.from_features == "":
            for i, phone in enumerate(phoneme_list):
                if i < before_len or i >= end_index_exclusive:
                    new_phoneme_list.append(deepcopy(phone))
                    continue

                before_environment = phoneme_list[i - before_len: i]
                after_environment = phoneme_list[i: i + after_len]

                matches_before = all([matches_features_dict(before_environment[j], before_features_list[j]) for j in range(before_len)])
                matches_after = all([matches_features_dict(after_environment[j], after_features_list[j]) for j in range(after_len)])
                if matches_before and matches_after:
                    new_phone = DEFAULT_FEATURE_VALUES
                    new_phone.update(self.to_features)
                    new_phoneme_list.append(new_phone)
                    new_phoneme_list.append(deepcopy(phone))
                else:
                    new_phoneme_list.append(deepcopy(phone))

        else:
            raise ValueError("invalid self.from_features: {0}".format(self.from_features))

        return LE2Word.from_phone_list([x for x in new_phoneme_list if x is not None])

    @staticmethod
    def get_random_sound_change_from_inventory(inventory):
        feature, value = get_random_feature_value_from_inventory(inventory)
        feature2, value2 = get_random_feature_value_from_inventory(inventory)
        feature3, value3 = get_random_feature_value_from_inventory(inventory)
        feature4, value4 = get_random_feature_value_from_inventory(inventory)
    
        from_features = {feature: value} if random.random() < 0.95 else ""
        to_features = {feature2: random.choice([i for i in PhoneticFeatureSpace.FEATURE_KEYS[feature2].keys()])} if random.random() < 0.95 else ""
        before_environment = {feature3: value3} if random.random() < 0.8 else WordBoundaryPhone().features
        after_environment = {feature4: value4} if random.random() < 0.8 else WordBoundaryPhone().features

        phonetic_environment = PhoneticEnvironment([before_environment], [after_environment])
        return SoundChange(from_features, to_features, phonetic_environment)
