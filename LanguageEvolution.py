import random

consonant_feature_defaults = {
    "type":"consonant",
    "voicing":"voiceless",
    "place":"alveolar",
    "manner":"plosive",
    "aspiration":"plain",
    "glottalization":"plain"
}

consonant_features = consonant_feature_defaults.keys()

vowel_feature_defaults = {
    "type":"vowel",
    "horizontal":"back",
    "vertical":"open",
    "roundedness":"unrounded"
}

vowel_features = vowel_feature_defaults.keys()

def shallow_copy(x):
    if type(x) is dict:
        return dict([(shallow_copy(k),shallow_copy(x[k])) for k in x])
    if type(x) is list:
        return [shallow_copy(i) for i in x]
    else:
        return x

class Phone:
    instances = []
    symbol_to_instance_dict = {"0": None}
    feature_values_string_to_symbol_dict = {}

    def __init__(self, consonant_or_vowel, feature_values_dict, ipa_symbol):
        if consonant_or_vowel not in ["consonant","vowel",None]:
            raise ValueError("Please select \"consonant\" or \"vowel\"")
        self.consonant_or_vowel = consonant_or_vowel
        self.feature_defaults = consonant_feature_defaults if consonant_or_vowel == "consonant" else vowel_feature_defaults
        self.features = consonant_features if consonant_or_vowel == "consonant" else vowel_features
        self.feature_values_dict = feature_values_dict
        for k in set(self.features) - set(feature_values_dict.keys()):
            self.feature_values_dict[k] = self.feature_defaults[k] # fill in default values for unspecified features

        self.feature_values_string = "_".join([k+"="+self.feature_values_dict[k] for k in sorted(self.features)])
        # print("feature_values_string for {0} is {1}".format(ipa_symbol,self.feature_values_string))
        # waste=input()
        if self.feature_values_string not in Phone.feature_values_string_to_symbol_dict:
            Phone.instances.append(self) # restrict it to here so the "copy" phone instances that get created aren't added
            Phone.feature_values_string_to_symbol_dict[self.feature_values_string] = ipa_symbol

        self.ipa_symbol = ipa_symbol
        if ipa_symbol not in Phone.symbol_to_instance_dict:
            Phone.symbol_to_instance_dict[ipa_symbol] = self

    def apply(self,_Rule):
        if _Rule is None:
            return self
        return _Rule.apply(self)


def get_all_feature_values_dict():
    d = {}
    for p in Phone.instances:
        for f in p.feature_values_dict:
            if f not in d:
                d[f] = [p.feature_values_dict[f]]
            else:
                if p.feature_values_dict[f] not in d[f]:
                    d[f].append(p.feature_values_dict[f])
    return d

def get_phone_from_symbol(symbol):
    if symbol in Phone.symbol_to_instance_dict:
        return Phone.symbol_to_instance_dict[symbol]
    else:
        return None

def get_symbol_from_feature_values_string(feature_values_string):
    if feature_values_string in Phone.feature_values_string_to_symbol_dict:
        result = Phone.feature_values_string_to_symbol_dict[feature_values_string]
    else:
        result = None
    # print("symbol of {0} is {1}".format(feature_values_string,result))
    return result


class Rule:
    instances = []

    def __init__(self,input_features_dict={},output_features_dict={},preceding_environment_features_dict={},following_environment_features_dict={},
        notation=""):
        Rule.instances.append(self)
        self.input_features_dict = input_features_dict
        self.input_features = input_features_dict.keys()
        self.input_feature_values = input_features_dict.values()

        self.output_features_dict = output_features_dict
        self.output_features = output_features_dict.keys()
        self.output_feature_values = output_features_dict.values()

        self.preceding_environment_features_dict = preceding_environment_features_dict
        self.preceding_environment_features = preceding_environment_features_dict.keys()
        self.preceding_environment_feature_values = preceding_environment_features_dict.values()

        self.following_environment_features_dict = following_environment_features_dict
        self.following_environment_features = following_environment_features_dict.keys()
        self.following_environment_feature_values = following_environment_features_dict.values()

        self.notation = notation

    def apply(self, _Phone, preceding_phone, following_phone):
        if self.input_features_dict == {} and self.output_features_dict == {} and self.notation != "":
            return self.apply_from_notation(_Phone,preceding_phone=preceding_phone,following_phone=following_phone)
        else:
            if _Phone.feature_values_dict["type"] not in self.input_features_dict["type"]:
                return _Phone # make no changes; the rule does not apply to the wrong type of sound
            d = shallow_copy(_Phone.feature_values_dict) # keep the unchanged values as what they originally were, not resetting to default
            for f in self.input_features:
                if _Phone.feature_values_dict[f] in self.input_features_dict[f]: # only apply rule if it, you know, applies
                    feature_value_index = self.input_features_dict[f].index(_Phone.feature_values_dict[f])
                    d[f] = self.output_features_dict[f][feature_value_index]

        return Phone(consonant_or_vowel=_Phone.feature_values_dict["type"],feature_values_dict=d,ipa_symbol=None)

    def apply_from_notation(self, _Phone, preceding_phone, following_phone):
        symbol = _Phone.ipa_symbol
        pb_symbol = preceding_phone.ipa_symbol if preceding_phone is not None else "#"
        pa_symbol = following_phone.ipa_symbol if following_phone is not None else "#"
        s = self.notation.replace(" ", "").split("/")
        change = s[0].split(">")
        change_from = change[0]
        change_to = change[1]

        if len(s) > 1:
            if "_" not in s[1]:
                raise ValueError("need underscore in environment string")
            t = s[1].split("_")
            if t[0] == "":
                preceding_environment = "0"
            elif t[0] == "#":
                preceding_environment = "#"
            else:
                preceding_environment = t[0]
            if t[1] == "":
                following_environment = "0"
            elif t[1] == "#":
                following_environment = "#"
            else:
                following_environment = t[1]
        else:
            preceding_environment = "0"
            following_environment = "0"

        # print("applying notation {0}, on {1}, in sequence {2}{1}{3}, with rule environment {4}_{5}".format(
        #     self.notation,symbol,pb_symbol,pa_symbol,preceding_environment,following_environment))
        # input()

        if matches_environment(symbol, change_from) and \
            matches_environment(pb_symbol,preceding_environment) and \
            matches_environment(pa_symbol,following_environment):
            # print("change made from {0} to {1}".format(symbol, change_to))
            result = change_to
        else:
            # print("keeping {0} as {0}".format(symbol))
            result = symbol

        return Phone.symbol_to_instance_dict[result]

def matches_environment(phone_symbol,environment_symbol):
    # print("seeing if {0} matches environment {1}".format(phone_symbol,environment_symbol))
    if phone_symbol == environment_symbol:
        return True
    if environment_symbol == "0":
        return True
    if environment_symbol in "VC":
        phone = get_phone_from_symbol(phone_symbol)
        cv1 = phone.consonant_or_vowel
        cv2 = "consonant" if environment_symbol == "C" else "vowel"
        return cv1 == cv2
    else:
        return False


def apply_sound_change(phones_list, _Rule):
    if type(phones_list) is str:
        raise
    phones_before_list = ["#"] + phones_list[:-1]
    phones_after_list = phones_list[1:] + ["#"]
    result = ""
    for i in range(len(phones_list)):
        _p = get_phone_from_symbol(phones_list[i])
        _pb = get_phone_from_symbol(phones_before_list[i])
        _pa = get_phone_from_symbol(phones_after_list[i])
        if _p is None:
            result_symbol = phones_list[i]
        else:
            new_p = _Rule.apply(_p, preceding_phone=_pb, following_phone=_pa)
            # print(new_p.ipa_symbol)
            if new_p is None:
                result_symbol = ""
            else:
                symbol = get_symbol_from_feature_values_string(new_p.feature_values_string)
                if symbol is None:
                    result_symbol = phones_list[i]
                else:
                    # print("appending symbol",symbol)
                    result_symbol = symbol
        result += result_symbol
        # print(result)
        # waste=input()

    return result.replace("0","")

def insert_random_phone(s):
    _p = random.choice(Phone.instances)
    symbol = _p.ipa_symbol

    i = random.choice(range(len(s)+1))
    if i == len(s):
        return s + [symbol]
    else:
        return s[:i]+[symbol]+s[i:]

def delete_random_phone(s):
    if len(s) <= 1:
        return s
    i = random.choice(range(len(s)))
    return s[:i]+s[i+1:]

def cleanup_repetitions(s,max_char_repetitions=2):
    result = []
    i = 0
    while i < len(s):
        si = s[i]
        j = i+1
        while j < len(s) and s[j] == si:
            j += 1
        if j-i > max_char_repetitions:
            result += [si]*max_char_repetitions
        else:
            result += s[i:j]
        i = j
    return result

def mutate(phones_list,n_times,deletion_probability=0.05,insertion_probability=0.05,max_char_repetitions=2):
    if type(phones_list) is str:
        raise
    result = phones_list
    for i in range(n_times):
        insert = random.random() < insertion_probability
        delete = random.random() < deletion_probability
        if insert or delete:
            if insert and delete:
                if random.random() < 0.5: # eliminate bias between inserting and deleting
                    result = insert_random_phone(result)
                else:
                    result = delete_random_phone(result)
            elif insert:
                result = insert_random_phone(result)
            elif delete:
                result = delete_random_phone(result)
        else:
            _r = random.choice(Rule.instances)
            result = apply_sound_change(result,_r)
    return cleanup_repetitions(result,max_char_repetitions=max_char_repetitions)

def collect_mutations(phones_list,n_times,repetitions_without_progress_tolerance=None,n_results=float("inf"),
    deletion_probability=0.05,insertion_probability=0.05,max_char_repetitions=2):
    # please note that this is not intended to be exhaustive!
    if repetitions_without_progress_tolerance is None:
        repetitions_without_progress_tolerance = n_times**2 # need a better bound for the tolerance
    result = []
    repetitions_without_progress = 0
    while repetitions_without_progress < repetitions_without_progress_tolerance and len(result) < n_results:
        m = mutate(phones_list,n_times,deletion_probability=deletion_probability,insertion_probability=insertion_probability,
            max_char_repetitions=max_char_repetitions)
        if m not in result:
            result.append(m)
            repetitions_without_progress = 0
        else:
            repetitions_without_progress += 1
    return sorted(result)

def mutate_vocabulary(str_lst,n_times,deletion_probability=0.05,insertion_probability=0.05,max_char_repetitions=2):
    index_order = [i for i in range(len(str_lst))]
    random.shuffle(index_order) # to eliminate bias about which words get priority for being the first to reach a new mutation
    str_lst = [str_lst[i] for i in index_order]

    result = str_lst

    for t in range(n_times):
        #new_result = []
        #for s in result:
            #new_s = None
            #while new_s is None or new_s in new_result:
                #new_s = mutate(s,1,deletion_probability=deletion_probability,insertion_probability=insertion_probability,
                    #max_char_repetitions=max_char_repetitions)
            #new_result.append(new_s)
        #result = new_result

        _r = random.choice(Rule.instances)
        for i in range(len(result)):
            result[i] = apply_sound_change(result[i],_r)
            result[i] = cleanup_repetitions(result[i],max_char_repetitions=max_char_repetitions)


    unshuffled_index_order = [index_order.index(i) for i in range(len(str_lst))]
    return [result[i] for i in unshuffled_index_order]




WordBoundaryPhone = Phone(consonant_or_vowel=None,feature_values_dict={"type":"WordBoundary"},ipa_symbol="#")

A = Phone(consonant_or_vowel="vowel",feature_values_dict={},ipa_symbol="a") # the default feature values will fill in
# print(A.feature_values_string)
E = Phone(consonant_or_vowel="vowel",feature_values_dict={"vertical":"mid"},ipa_symbol="e")
I = Phone(consonant_or_vowel="vowel",feature_values_dict={"vertical":"close"},ipa_symbol="i")
O = Phone(consonant_or_vowel="vowel",feature_values_dict={"vertical":"mid","roundedness":"rounded"},ipa_symbol="o")
U = Phone(consonant_or_vowel="vowel",feature_values_dict={"vertical":"close","roundedness":"rounded"},ipa_symbol="u")
Y = Phone(consonant_or_vowel="vowel",feature_values_dict={"vertical":"close","roundedness":"rounded","horizontal":"front"},ipa_symbol="y")

M = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"nasal","place":"bilabial","voicing":"voiced"},ipa_symbol="m")
N = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"nasal","place":"alveolar","voicing":"voiced"},ipa_symbol="n")

P = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"bilabial"},ipa_symbol="p")
B = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"bilabial","voicing":"voiced"},ipa_symbol="b")
T = Phone(consonant_or_vowel="consonant",feature_values_dict={},ipa_symbol="t")
D = Phone(consonant_or_vowel="consonant",feature_values_dict={"voicing":"voiced"},ipa_symbol="d")
C = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"palatal"},ipa_symbol="c")
K = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"velar"},ipa_symbol="k")
G = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"velar","voicing":"voiced"},ipa_symbol="g")
Q = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"uvular"},ipa_symbol="q")

F = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"fricative","place":"labiodental"},ipa_symbol="f")
V = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"fricative","place":"labiodental","voicing":"voiced"},ipa_symbol="v")
S = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"fricative"},ipa_symbol="s")
Z = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"fricative","voicing":"voiced"},ipa_symbol="z")
X = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"fricative","place":"velar"},ipa_symbol="x")
H = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"fricative","place":"glottal"},ipa_symbol="h")

L = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"lateral-approximant","voicing":"voiced"},ipa_symbol="l")
R = Phone(consonant_or_vowel="consonant",feature_values_dict={"manner":"trill","voicing":"voiced"},ipa_symbol="r")
J = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"palatal","manner":"approximant","voicing":"voiced"},ipa_symbol="j")
W = Phone(consonant_or_vowel="consonant",feature_values_dict={"place":"bilabial","manner":"approximant","voicing":"voiced"},ipa_symbol="w")


# NoChangeRule = None  # just use None instead of "typed Nones"

Raising = Rule(
    input_features_dict={"type":["vowel"],"vertical":["open","mid"]},
    output_features_dict={"type":["vowel"],"vertical":["mid","close"]}
)

Lowering = Rule(
    input_features_dict={"type":["vowel"],"vertical":["mid","close"]},
    output_features_dict={"type":["vowel"],"vertical":["open","mid"]}
)

Voicing = Rule(
    input_features_dict={"type":["consonant"],"voicing":["voiceless"]},
    output_features_dict={"type":["consonant"],"voicing":["voiced"]}
)

Devoicing = Rule(
    input_features_dict={"type":["consonant"],"voicing":["voiced"]},
    output_features_dict={"type":["consonant"],"voicing":["voiceless"]}
)

Rounding = Rule(
    input_features_dict={"type":["vowel"],"roundedness":["unrounded"]},
    output_features_dict={"type":["vowel"],"roundedness":["rounded"]}
)

Derounding = Rule(
    input_features_dict={"type":["vowel"],"roundedness":["rounded"]},
    output_features_dict={"type":["vowel"],"roundedness":["unrounded"]}
)


SomeRule = Rule(notation="C > 0 / _#")
# print(apply_sound_change([i for i in "kanakekektket"], SomeRule))

# print(Raising.apply(A).feature_values_string)
# print(A.apply(Raising).feature_values_string) # should be the same
# print(K.apply(None).feature_values_string)

# print(get_all_feature_values_dict())

# print(apply_sound_change("iea",Rounding))
# print(apply_sound_change("kakate tenini",Raising))
# print(apply_sound_change("kakate tenini",Lowering))
# print(apply_sound_change("kakatetenini",SomeRule))

# print("".join(mutate([i for i in "kakatetenini"],1)))
# print(collect_mutations([i for i in "kakatetenini"],100,repetitions_without_progress_tolerance=10,n_results=4,
#     deletion_probability=0.1,insertion_probability=0.1,max_char_repetitions=2))
# interestingly, the length of the resulting list decreases after n_times=6, probably due to distribution and constant tolerance

devonian_vocabulary = ['aek', 'ah', 'ahe', 'ahn', 'aka', 'amh', 'amn', 'an', 'ans', 'ar', 'ats', 'ea', 'eah', 'eik', 'eil', 'ele', 'ema', 'emi', 
                       'ems', 'era', 'ere', 'esn', 'eti', 'etk', 'h', 'han', 'het', 'hih', 'hki', 'hs', 'hsk', 'hti', 'i', 'ien', 'ika', 'ikt', 
                       'il', 'ila', 'imi', 'int', 'ish', 'isl', 'kan', 'kar', 'keh', 'kek', 'kh', 'khs', 'kil', 'klt', 'kma', 'kme', 'kna', 'kre', 
                       'krk', 'kst', 'lem', 'lie', 'lkl', 'lre', 'm', 'ma', 'mai', 'mat', 'mei', 'mes', 'mie', 'mk', 'mra', 'mti', 'n', 'nat', 'nin', 
                       'nk', 'nka', 'nlk', 'nme', 'nse', 'ran', 're', 'rnk', 'rtk', 's', 'sam', 'sel', 'shr', 'shs', 'sia', 'ske', 'skh', 'sl', 'sli', 
                       'sm', 'sml', 'str', 't', 'tah', 'tai', 'tas', 'tat', 'tem', 'ten', 'tk', 'tmt', 'tn', 'trt', 'tsa', 'tst']
#new_vocabulary = mutate_vocabulary(devonian_vocabulary,10)
#print([(devonian_vocabulary[i],new_vocabulary[i]) for i in range(len(devonian_vocabulary))])


# possibly should create new module here, so I can split up the language evolution parts into phonology and syntax (later will do morphology)

# treat sentences as if S, O, and V are completely separate parts, don't worry about expressing things like V(O, S)
# sentence = "((man def) nom) (apel ak) (it pas)"
# here all branching is binary with the operator following its scope (if there is only one word in the scope I omit parentheses)
sentence = {
    "S": {
        "NOM": {
            "DEF": "man"
        }
    },
    "O": {
        "ACC": "apple"
    },
    "V": {
        "PAST": "eat"
    }
}
# for purposes of marking, the computer should know that "man" is S, NOM, and DEF, that DEF is S and NOM, and (although this is assumed) that NOM is S
# desired output for glosslang: "man def nom apel ak it pas"
# desired output for Proto-Contic: "tse-ma bahi-ri ze-do-gi"
# desired output for German: "der Mann ass den Apfel" (don't worry about conversational past for now)

