# modification of Levenshtein distance to include changes in feature matrices
# each feature must have a distance metric; categorical features can be discrete metric (d(x,y) = 1 iff x != y); can also have scalar features like degree of constriction, position of tongue blade, etc.
# this will create a distance metric between segments in articulatory space (although not in acoustic space), and could help in cognate detection

import random
import numpy as np
import math


class Feature:
    def __init__(self, name, metric):
        self.name = name
        assert callable(metric)
        self.metric = metric

    discrete_metric = lambda x,y: 0 if x == y else 1
    scalar_metric = lambda x,y: abs(x-y)


class FeatureValueObject:
    def __init__(self, dct, features):
        dct_feature_names = dct.keys()
        feature_names = [f.name for f in features]
        assert all(n in feature_names for n in dct_feature_names), "need a Feature object for all feature names in dct keys"
        self.name_to_value = dct
        self.name_to_feature = {f.name: f for f in features}
        self.defined_feature_names = set(dct_feature_names)

    def has_feature(self, name):
        return name in self.defined_feature_names

    def __getitem__(self, index):
        return self.name_to_value.get(index)

    def distance(self, other, weights=None):
        if type(other) is not FeatureValueObject:
            return NotImplemented
        all_feature_names = self.defined_feature_names | other.defined_feature_names
        d = 0
        # print("calculating distance")
        for n in all_feature_names:
            self_val = self[n]
            other_val = other[n]
            feat = get_feature_object(n, [self, other])
            metric = feat.metric
            dd = metric(self_val, other_val)
            assert dd == metric(other_val, self_val), "asymmetric metric"
            if weights is not None:
                assert n in weights, f"need weight for feature {n}"
                w = weights[n]
                weighted_dd = dd * w
            else:
                w = 1
                weighted_dd = dd
            # print(f"feature {n} has values {self_val} and {other_val}, with distance {dd}, weighted by factor {w} to {weighted_dd}")
            d += weighted_dd
            # print(f"distance is now {d}")
        return d

    def __repr__(self):
        return "FVO" + repr(self.name_to_value)


def get_feature_object(name, fv_objs):
    res = None
    for fv in fv_objs:
        feat = fv.name_to_feature.get(name)
        if feat is not None:
            if res is None:
                res = feat
            else:
                assert feat == res, f"feature conflict with name {name}"
    return res



if __name__ == "__main__":
    f_color = Feature("color", Feature.discrete_metric)
    f_letter = Feature("letter", Feature.discrete_metric)
    f_age = Feature("age", Feature.scalar_metric)
    f_height = Feature("height", Feature.scalar_metric)

    def get_random_obj():
        d = {
            "color": random.choice(["red", "yellow", "blue"]),
            "letter": random.choice("ABCDEFG"),
            "age": random.randint(1, 100),
            "height": random.randint(8, 14) / 2,
        }
        features = [f_color, f_letter, f_age, f_height]
        return FeatureValueObject(d, features)

    o1 = get_random_obj()
    o2 = get_random_obj()
    weights = {n: random.randint(1, 10) for n in o1.defined_feature_names}
    # for sound change of a single sound, want to learn what these feature weights are (if there is a consistent set of them, it might be non-linear with interactions)
    # basically want estimators for the probability of any given sound change (and note that asymmetry of directionality exists)
    # 

    print(f"weights: {weights}")
    print(o1)
    print(o2)
    print(o1.distance(o2, weights))
    print(o2.distance(o1, weights))

