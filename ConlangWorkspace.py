# coding=utf-8

secret_key = 0

# BUGS
"""
- FIXED - adjectives left out in translations (3275104 to 73; 1 to 2)
- FIXED - adjective-noun ordering is the reverse in example sentences from what is
said at the top (3275104, 73)
- FIXED - list index out of range on translate when pos is "?"
- MEH TOO LAZY - general messiness in exception handling (not really a bug)
- still happening - roots in lexicon display forms that never surface (e.g. in a lang where
uu > u always, the root "ruum" should be displayed in the lexicon as "rum")
- program exits on certain errors (my fault) instead of just stopping that
command and doing the loop again (I just need to learn how to do this)
- -d=z is changed to -r=z in some languages (just need to restrict that
morphophonemic rule somehow, and prevent similar things in the future)
- translating to or from seeds with large absolute values gives a completely
different language and thus returns errors with high likelihood
(this is a bug with vet_seed, switch_sign, and the like. check also translate)
- infinite loop rarely (seed 3842282112524023590), with "(" in parse_structure
"""

# TO DO
"""
- add verb / sentence negation and possibly other morphemes
- add more affix categories (infix, isolating) and variability of affix type
by morpheme (e.g. "non parlerò", "ar v-nax-av-t") (will require lots of restructuring to
change the "prefix" truth value to an enumerable (does not actually have to
be enum, but should fall within a global set that can be a declared list))
- maybe allow for smaller affixes, but this is not that important
- print out morphophonemic and phonological rules; people won't have to
use them for composition, but they will receive messages in collapsed form
in their own language
"""

# auxiliary functionality

def r():
    for i in range(secret_key):
        # waste a random value
        a = random.random()
    return random.random()

def intable(s):
    """
    Returns True if a string can be converted to an integer,
    False otherwise.
    """
    try:
        int(s)
        return True
    except:
        return False

def list_nested_dict(d):
    """
    Takes a dictionary nested to arbitrarily many dimensions,
    returns a list of all the values at the ends of paths,
    i.e. those values which are not dicts.
    """
    result = []
    for key in sorted(d):
        if type(d[key]) == dict:
            result.extend(list_nested_dict(d[key]))
        else:
            result.append(d[key])
    return result

def list_locations(list_of_lists):
    """
    Takes a list of lists, returns a dict with lists of the indices
    of lists in which each item appears. Used for union and intersection.
    """
    locations = {}
    i = -1
    for lst in list_of_lists:
        i += 1
        for e in lst:
            if e in locations and i not in locations[e]:
                # prevents double-counting of items that appear more than once
                # in the same list
                locations[e].append(i)
            else:
                locations[e] = [i]
    return locations

def union(list_of_lists):
    """
    Takes a list of lists, returns a list of the items found in any list.
    """
    result = []
    locations = list_locations(list_of_lists)
    for u in sorted(locations):
        if len(locations[u]) > 0:
            result.append(u)
    return result

def intersection(list_of_lists):
    """
    Takes a list of lists, returns a list of the items found in all lists.
    """
    result = []
    locations = list_locations(list_of_lists)
    for u in sorted(locations):
        if locations[u] == [i for i in range(len(list_of_lists))]:
            result.append(u)
    return result

def partial_intersection(list_of_lists, n):
    """
    Takes a list of lists and an integer,
    returns a list of the items found in at least n of the lists.
    """
    result = []
    locations = list_locations(list_of_lists)
    for u in sorted(locations):
        if len(locations[u]) >= n:
            result.append(u)
    return result    

def prob(x, y, seed):
    """
    Takes two integers x and y, returns True with probability x/y.
    """
    random.seed(seed)
    return random.randrange(y) < x

def remove_duplicates(lst):
    result = []
    for el in lst:
        if el not in result:
            result.append(el)
    return result

def add_pos(word, pos):
    entry = word + " ("+ pos_to_letter[pos] + ")"
    return entry

def remove_pos(entry):
    if "(" in entry:
        i = entry.index("(")
        word = entry[:i-1] # excludes the space before the parenthesis
    else:
        word = entry
    return word

def choose(d, seed):
    """
    Takes a dict of entries and relative probabilities,
    chooses one of them. Probabilities need not add to 1.
    """
    random.seed(seed)
    t = (sum(d.values()))
    sel = t * random.random()
    for key in sorted(d):
        sel -= d[key]
        if sel <= 0:
            return key    

def vet_seed(s, seed):
    random.seed(seed)
    if intable(s):
        j = int(s)
        j = j % sys.maxsize
    else:
        j = random.randrange(-sys.maxsize/2, sys.maxsize/2+1)
        # note that ints in Python 3 have no maximum,
        # so these bounds could be anything
    return j

def truncnorm(mu, sigma, lower, upper, seed):
    random.seed(seed)
    result = random.normalvariate(mu, sigma)
    result = max(result, lower)
    result = min(result, upper)
    return result

def switch_sign(s):
    if type(s) != int:
        return s
    if s < sys.maxsize - s:
        t = s
    else:
        t = s - sys.maxsize
    return t

dead_keys = ["=", "<", ">", "^"]
ipa_symbols = {
    ">n":"ŋ",
    "^h":"ʰ",
    "=s":"ʃ",
    "=z":"ʒ",
    "=g":"ɣ",
    ">o":"ø",
    "<a":"æ"
}

def render_ipa(s):
    result = ""
    i = 0
    while i < len(s):
        if s[i] in dead_keys:
            result += ipa_symbols[s[i:i+2]]
            i += 2
        else:
            result += s[i]
            i += 1
    return result

# phonology functionality

C_moas = ["nasal", "plosive", "fricative", "affricate", "approximant"]
C_poas = ["labial", "coronal", "dorsal"]
C_voicing = ["voiceless", "voiced", "aspirated"]

V_backness = ["front", "back"]
V_height = ["close", "mid", "open"]
V_roundedness = ["unrounded", "rounded"]

C_book = {
    "nasal" : {
        "labial" : {
            "voiceless" : None,
            "voiced" : "m",
            "aspirated" : None
        },
        "coronal" : {
            "voiceless" : None,
            "voiced" : "n",
            "aspirated" : None
        },
        "dorsal" : {
            "voiceless" : None,
            "voiced" : "ŋ",
            "aspirated" : None
        }
    },
    "plosive" : {
        "labial" : {
            "voiceless" : "p",
            "voiced" : "b",
            "aspirated" : "pʰ"
        },
        "coronal" : {
            "voiceless" : "t",
            "voiced" : "d",
            "aspirated" : "tʰ"
        },
        "dorsal" : {
            "voiceless" : "k",
            "voiced" : "g",
            "aspirated" : "kʰ"
        }
    },
    "fricative" : {
        "labial": {
            "voiceless" : "f",
            "voiced" : "v",
            "aspirated": None
        },
        "coronal" : {
            "voiceless" : "s",
            "voiced" : "z",
            "aspirated" : None
        },
        "dorsal" : {
            "voiceless" : "x",
            "voiced" : "ɣ",
            "aspirated" : "h"
        }
    },
    "affricate" : {
        "labial" : {
            "voiceless" : None,
            "voiced" : None,
            "aspirated" : None
        },
        "coronal" : {
            "voiceless" : "tʃ",
            "voiced" : "dʒ",
            "aspirated" : "tʃʰ"
        },
        "dorsal" : {
            "voiceless" : "kx",
            "voiced" : None,
            "aspirated" : None
        }
    },
    "approximant" : {
        "labial" : {
            "voiceless" : None,
            "voiced" : "w",
            "aspirated" : None
        },
        "coronal" : {
            "voiceless" : None,
            "voiced" : "r",
            "aspirated" : None
        },
        "dorsal" : {
            "voiceless" : None,
            "voiced" : "j",
            "aspirated" : None
        }
    }
}

V_book = {
    "front" : {
        "close" : {
            "unrounded" : "i",
            "rounded" : "y"
        },
        "mid" : {
            "unrounded" : "e",
            "rounded" : "ø"
        },
        "open" : {
            "unrounded" : "æ"
        }
    },
    "back" : {
        "close" : "u",
        "mid" : "o",
        "open" : "a"
    }
}

phonological_rulesets = {
    "long_vowel_diphthongization":{"aa":"aj", "ee":"ej", "ei":"ej", "ii":"ij", "oo":"ow", "ou":"ow", "uu":"uw"},
    "long_vowel_monophthongization":{"aa":"a", "ee":"e", "ej":"e", "ei":"e", "ii":"i", "ij":"i", "oo":"o", "ow":"o", "ou":"o", "uu":"u", "uw":"u"}
}

""" # don't need anymore?
def list_inventory(categorized_inventory, dimensions = 1):
    workspace = list(categorized_inventory) # list of the keys
    for i in range(dimensions):
        key_list = workspace
        workspace = []
        for key in key_list:
            workspace.append(key_list[key])
            
    #for category in categorized_inventory:
        #for i in categorized_inventory[category]:
            #inventory.append(i)
    inventory = workspace
    print(inventory)
    return inventory
"""

def phonemes_with_features(d, features):
    """
    Takes a feature dictionary (C_book or V_book)
    and a list of features, e.g., ["plosive", "coronal"],
    returns a list of the phonemes in that dict with those features.
    This is an AND operator. The example above will give ["t", "d", "th"].
    """
    counts = {}
    findings = []
    result = []
    for feature in sorted(features):
        if feature in d: # features on the same level of the dicts are MX
            if type(d[feature]) == dict:
                findings.extend(phonemes_with_features(d[feature], features))
            elif d[feature] != None:
                findings.append(d[feature]) # d[feature] is the phoneme
        else:
            for key in sorted(d):
                if type(d[key]) == dict:
                    findings.extend(phonemes_with_features(d[key], features))
                elif d[key] != None:
                    findings.append(d[key])
    for u in findings:
        if u in counts:
            counts[u] += 1
        else:
            counts[u] = 1
    for phoneme in sorted(counts):
        if counts[phoneme] > len(features):
            print("Error: things are being counted more than once per feature")
        elif counts[phoneme] == len(features):
            result.append(phoneme)
    return result

def gen_features(c_or_v, seed):
    """
    Takes a type, "C" or "V".
    Selects a group of features for use in creating the phoneme inventory.
    """
    random.seed(seed)
    features = []
    if c_or_v == "C":
        features.extend(["plosive", "nasal"])
        for i in C_moas:
            if prob(2,3, r()) and i not in features:
                features.append(i)
        for i in C_poas:
            if prob(2,3, r()) and i not in features:
                features.append(i)
        for i in C_voicing:
            if prob(2,3, r()) and i not in features:
                features.append(i)
    if c_or_v == "V":
        for i in V_backness:
            if prob(1,1, r()) and i not in features:
                features.append(i)
        for i in V_height:
            if prob(1,1, r()) and i not in features:
                features.append(i)
        for i in V_roundedness:
            if prob(1,1, r()) and i not in features:
                features.append(i)
    return features

def gen_inventory(categories, seed):
    """
    Takes a category list, generates a phoneme inventory,
    returns it formatted by category (but not feature).
    """
    random.seed(seed)
    categorized_inventory = {} # dict of lists
    for category in sorted(categories): # sorted() prevents random() bug
        categorized_inventory[category] = []
        while categorized_inventory[category] == [] and categories[category] != []:
            for i in categories[category]:
                if prob(1, 1, r()):
                    categorized_inventory[category].append(i)
    return categorized_inventory

def gen_syllable_structure(categorized_inventory, seed):
    # format: onset-nucleus-coda
    # within onset: initial|other
    # within coda: other|final
    # notation: "1|2-3-4|5"
    # example: "C(R)|C-V-(C)|(N)"
    # initial = "C(R)V(C)"; medial = "CV(C)"; final = "CV(N)"
    random.seed(seed)
    
    p1 = choose({"C":2, "(C)":1, "C(R)":1, "":1}, r())
    p2 = "C" + choose({"(C)":1, "(R)":1, "":4}, r())
    p3 = "V" + choose({"(V)":1, "":5}, r())
    p4 = choose({"C":1, "(C)":3, "(R)":2, "(N)":3, "":5}, r())
    p5 = choose({"C":1, "(C)":2, "(N)":2, "":7}, r())

    full_structure = p1 + "|" + p2 + "-" + p3 + "-" + p4 + "|" + p5
    return full_structure

def parse_structure(full_structure, position):
    """
    Takes the full syllable structure and creates the individual
    syllable structure for the position ("initial", "medial", "final",
    or "peripheral").
    """
    onset = full_structure.split("-")[0].split("|")[1]
    nucleus = full_structure.split("-")[1]
    coda = full_structure.split("-")[2].split("|")[0]
    if position == "initial" or position == "peripheral":
        onset = full_structure.split("-")[0].split("|")[0]
    if position == "final" or position == "peripheral":
        coda = full_structure.split("-")[2].split("|")[1]
    if position == "medial":
        # don't change the defaults
        pass

    structure = onset + nucleus + coda
    return structure

def gen_syllable(categorized_inventory, structure, seed):
    """
    Takes a categorized inventory, such as that produced by gen_inventory(),
    a string that represents a syllable structure, and the syllable position,
    returns a random syllable that is viable given these parameters.
    """
    random.seed(seed)
    
    syllable = ""
    i = 0
    while i < len(structure):
        advance = 0
        if structure[i] in sorted(categorized_inventory):
            syllable = syllable + categorized_inventory[structure[i]][random.randrange(len(categorized_inventory[structure[i]]))]
            advance = 1
        elif structure[i] == "(":
            opens = 1
            optional = ""
            j = i+1
            #if structure[j] == "(":
                #for k in range(j, len(structure)):
                    #if structure[k] == ")":
                        #optional = optional + structure[j]
                        #j += 1
                        #break
                #if prob(1, 2):
                    #syllable = syllable + gen_syllable(categorized_inventory, optional)
            while j < len(structure) and opens > 0:
                # currently assuming no nested parentheses
                if structure[j] == "(":
                    opens += 1
                elif structure[j] == ")":
                    opens -= 1
                if opens != 0:
                    optional = optional + structure[j]
                    j += 1
            advance = j-i+1
            if random.randrange(2) == 0:
                syllable = syllable + gen_syllable(categorized_inventory, optional, r())
        i += advance
    return syllable

def gen_word(categorized_inventory, full_structure, num_syllables, seed):
    random.seed(seed)
    word = ""
    if num_syllables < 1:
        print("You have to have at least one syllable!")
        return None
    if num_syllables == 1:
        word = gen_syllable(categorized_inventory, parse_structure(full_structure, "peripheral"), r())
    else:
        for i in range(num_syllables):
            if i == 0:
                word = word + gen_syllable(categorized_inventory, parse_structure(full_structure, "initial"), r())
            elif i == num_syllables - 1:
                word = word + gen_syllable(categorized_inventory, parse_structure(full_structure, "final"), r())
            else:
                word = word + gen_syllable(categorized_inventory, parse_structure(full_structure, "medial"), r())
    return word

# morphology functionality

def affix(word, key, function, prefix):
    """
    Takes a word, the function of the affix, and a truth value for prefixing.
    Returns the word with the affix added.
    """
    affix = key["function"][function]
    if prefix:
        return affix + "-" + word
    else:
        return word + "-" + affix

def collapse(gloss, pre_rules, post_rules):
    """
    Takes a string of words separated into morphemes,
    collapses them into surface forms.
    """
    result = gloss
    for rank in sorted(pre_rules):
        for key in sorted(pre_rules[rank]):
            if key in result:
                result = result.replace(key, pre_rules[rank][key])
    result = result.replace("-", "")
    for rank in sorted(post_rules):
        for key in sorted(post_rules[rank]):
            if key in result:
                result = result.replace(key, post_rules[rank][key])    
    
    """
    if prob(1, 2, r()):
        # remove double letters (naive because of digraphs, but whatevs)
        new_result = ""
        for i in range(len(result)-1):
            if result[i] != result[i+1]:
                new_result += result[i]
        new_result += result[-1]
        result = new_result
    """
    return result

# syntax functionality

def gen_word_order(seed):
    """
    Chooses one of the six word orders with similar probability to IRL.
    Does not support others, such as V2.
    """
    random.seed(seed)
    
    # realistic weights
    #return choose({"SOV": 0.45, "SVO": 0.42, "VSO": 0.09, "VOS": 0.03, "OVS": 0.005, "OSV": 0.005}, r())

    # give some more love to the underdogs
    return choose({"SOV": 0.25, "SVO": 0.25, "VSO": 0.15, "VOS": 0.15, "OVS": 0.1, "OSV": 0.1}, r())

def infer(roots, key, pos_profile, word_order, adjective_noun):
    if "X" in pos_profile:
        return infer(roots, key, pos_profile.replace("X", ""), word_order, adjective_noun)

    l = len(pos_profile)
    possibilities = ["This string should never get returned."]
    if adjective_noun:
        seg = "AN"
    else:
        seg = "NA"
    if l == 3:
        return word_order.replace("S", "N").replace("O", "N")
    elif l == 4:
        possibilities = [word_order.replace("S", seg).replace("O", "N"), word_order.replace("S", "N").replace("O", seg)]
    elif l == 5:
        return word_order.replace("S", seg).replace("O", seg)

    acceptables = []
    for p in possibilities:
        use_it = True
        for pos_i in range(len(p)):
            if roots[pos_i] not in key[letter_to_pos[p[pos_i]]]:
                use_it = False
                break
        if use_it:
            acceptables.append(p)
    if len(acceptables) == 0:
        print("There were no possible readings of this sentence.")
        sys.exit()
    if len(acceptables) != 1:
        print("There were {0} possible readings of this sentence.\n"
              "The reading {1} was used.".format(len(acceptables), acceptables[0]))
    #print("acceptables:", acceptables)
    return acceptables[0]

    # GRAVEYARD OF IMPLEMENTATION ATTEMPTS
    # THEY WERE SO YOUNG

    """
    if word_order[0] == "V":
        if pos_profile[0] == "?":
            pos_profile = "V" + pos_profile[1:]
    """

    """
    if adjective_noun:
        max_profile = word_order.replace("S", "AN").replace("O", "AN")
    else:
        max_profile = word_order.replace("S", "NA").replace("O", "NA")
    """
    
    """
    q = 0
    for i in pos_profile:
        if i == "?":
            q += 1
    if q == 0:
        return pos_profile
    """

    """
    if adjective_noun:
        ender = "N"
    else:
        ender = "A"
    x = pos_profile.index("?")
    """
    
    """
    if q == 1:
        if "V" not in pos_profile:
            return pos_profile.replace("?", "V")
        elif adjective_noun:
            if
    """

def parse(pos_profile, word_order, adjective_noun):
    """
    Given a string representing the parts of speech of the words
    in a sentence, the word order, and the truth value for AN,
    outputs a dict of dicts telling the index of each word in the syntax.
    This description is badly worded, so look at this:
    Example input: ("ANVN", SVO, True)
    Resulting output: {
        "subj": {"noun": 1, "adjective": 0},
        "obj": {"noun": 3},
        "verb": {"verb": 2}
    }
    """
    result = {"subj": {}, "obj": {}, "verb": {}}
    verb_location = pos_profile.index("V")
    result["verb"]["verb"] = verb_location
    if adjective_noun:
        adjective_to_noun = -1
    else:
        adjective_to_noun = 1
    noun_locations = [m.start() for m in re.finditer("N", pos_profile)]
    #print(noun_locations)
    first_noun_location = noun_locations[0]
    first_adjective_location = -1
    try:
        a = pos_profile[first_noun_location + adjective_to_noun]
        if a == "A":
            first_adjective_location = first_noun_location + adjective_to_noun
    except:
        pass
    second_noun_location = noun_locations[1]
    second_adjective_location = -1
    try:
        a = pos_profile[second_noun_location + adjective_to_noun]
        if a == "A":
            second_adjective_location = second_noun_location + adjective_to_noun
    except:
        pass
    if word_order.index("S") < word_order.index("O"):
        np1 = "subj"
        np2 = "obj"
    else:
        np1 = "obj"
        np2 = "subj"
    result[np1]["noun"] = first_noun_location
    if first_adjective_location != -1:
        result[np1]["adjective"] = first_adjective_location
    result[np2]["noun"] = second_noun_location
    if second_adjective_location != -1:
        result[np2]["adjective"] = second_adjective_location
    #print(result)
    return result

# lexicon functionality

pos_to_letter = {"noun" : "N", "verb" : "V", "adjective" : "A", "function" : "F"}
letter_to_pos = {"N" : "noun", "V" : "verb", "A" : "adjective", "F" : "function"}

V_functions = ["past"]
N_functions = ["plural"]

eng_lex = {
    "noun" : ["man", "woman", "boy", "girl", "cat", "dog", "tree", "car", "house", "ship", "guitar", "gun", "shoe", "book", "cake"],
    "verb" : ["see", "eat", "want", "like", "have", "find", "need", "buy"],
    "adjective" : ["good", "bad", "tall", "short", "big", "small", "red", "blue", "black", "white", "green", "yellow", "old", "new"],
    "function" : [i for i in V_functions] + [i for i in N_functions]
}

def gen_NP(lex, seed):
    """
    Takes a lexicon, returns a dict with the NP information.
    """
    random.seed(seed)
    NP = {}
    if prob(1, 2, r()):
        NP["adjective"] = lex["adjective"][random.randrange(len(lex["adjective"]))]
    NP["noun"] = lex["noun"][random.randrange(len(lex["noun"]))]
    NP["plural"] = (random.randrange(2) == 0)
    return NP

def gen_VP(lex, seed):
    """
    Takes a lexicon, returns a dict with the VP information.
    """
    random.seed(seed)
    VP = {}
    VP["verb"] = lex["verb"][random.randrange(len(lex["verb"]))]
    VP["past"] = prob(1, 2, r())
    return VP

def gen_sentence(lex, seed):
    """
    Takes a lexicon.
    Returns a dict of "subj", "obj", and "verb" mapping to
    the corresponding constituents (dicts of features).
    """
    random.seed(seed)
    subj = gen_NP(lex, r())
    obj = gen_NP(lex, r()) # "object" is a keyword in Python
    verb = gen_VP(lex, r())
    return {"subj":subj, "obj":obj, "verb":verb}

# should probably remove this function in favor of using
# render_lang for everything including English
# WARNING: this function was creating global vars for adjective_noun, etc.
"""
def render_eng(sentence):
    #"#"#"
    #Creates an English sentence from the sentence structure.
    #Currently supports only rudimentary caveman speak.
    #"#"#"
    local_sentence_words = []
    local_word_order = "SVO"
    local_adjective_noun = True
    for i in local_word_order:
        if i == "S":
            noun = sentence["subj"]["noun"]
            if sentence["subj"]["plural"]:
                noun = noun + "-s"
            if "adjective" in sentence["subj"]:
                adjective = sentence["subj"]["adjective"]
                if local_adjective_noun:
                    local_sentence_words.append(adjective)
                    local_sentence_words.append(noun)
                else:
                    local_sentence_words.append(adjective)
                    local_sentence_words.append(noun)
            else:
                local_sentence_words.append(noun)
        elif i == "O":
            noun = sentence["obj"]["noun"]
            if sentence["obj"]["plural"]:
                noun = noun + "-s"
            if "adjective" in sentence["obj"]:
                adjective = sentence["obj"]["adjective"]
                if local_adjective_noun:
                    local_sentence_words.append(adjective)
                    local_sentence_words.append(noun)
                else:
                    local_sentence_words.append(adjective)
                    local_sentence_words.append(noun)
            else:
                local_sentence_words.append(noun)
        elif i == "V":
            verb = sentence["verb"]["verb"]
            if sentence["verb"]["past"]:
                verb = verb + "-ed"
            local_sentence_words.append(verb)
    result = " ".join(sentence_words)
    return result
"""

def render_lang(sentence, key, word_order, adjective_noun, prefix):
    """
    Creates a sentence in the conlang from the sentence structure.
    """
    ### DOES NOT DISTINGUISH BETWEEN HOMOPHONES
    # this is due to its use of eng_to_lang as key, ignoring the pos
    # the key should be sorted by the pos
    sentence_words = []
    for i in word_order:
        if i == "S":
            noun = key["noun"][sentence["subj"]["noun"]]
            if sentence["subj"]["plural"]:
                noun = affix(noun, key, "plural", prefix)
            if "adjective" in sentence["subj"]:
                adjective = key["adjective"][sentence["subj"]["adjective"]]
                if adjective_noun:
                    sentence_words.append(adjective)
                    sentence_words.append(noun)
                else:
                    sentence_words.append(noun)
                    sentence_words.append(adjective)
            else:
                sentence_words.append(noun)
        elif i == "O":
            noun = key["noun"][sentence["obj"]["noun"]]
            if sentence["obj"]["plural"]:
                noun = affix(noun, key, "plural", prefix)
            if "adjective" in sentence["obj"]:
                adjective = key["adjective"][sentence["obj"]["adjective"]]
                if adjective_noun:
                    sentence_words.append(adjective)
                    sentence_words.append(noun)
                else:
                    sentence_words.append(noun)
                    sentence_words.append(adjective)
            else:
                sentence_words.append(noun)
        elif i == "V":
            verb = key["verb"][sentence["verb"]["verb"]]
            if sentence["verb"]["past"]:
                verb = affix(verb, key, "past", prefix)
            sentence_words.append(verb)
    result = " ".join(sentence_words)
    return result

def translate(message, f, t):
    Lf = L(f)
    Lt = L(t)

    sentence = {"subj":{}, "obj":{}, "verb":{}}

    """
    for i in Lf.word_order:
        if i == "V":
            #constituents.append("VP")
            pass
        elif i == "S" or i == "O":
            #constituents.append("NP")
            pass
    """

    message = render_ipa(" ".join(message)).split(" ")

    pos_profile = ""
    for word in message:
        still_looking = True
        best_pos = "?"
        morphemes = word.split("-")
        i = 0
        while still_looking and i < len(morphemes):
            possibilities = {}
            for pos in Lf.lang_lex:
                if morphemes[i] in Lf.lang_lex[pos]:
                    possibilities[pos]=Lf.lang_to_eng[pos][morphemes[i]]
            #print(morphemes[i], possibilities)
            if len(possibilities) == 1:
                if "function" in possibilities:
                    if possibilities["function"] in V_functions:
                        best_pos = "V"
                        still_looking = False
                    elif possibilities["function"] in N_functions:
                        best_pos = "N"
                        still_looking = False
                elif "verb" in possibilities:
                    best_pos = "V"
                    still_looking = False
                elif "noun" in possibilities:
                    best_pos = "N"
                    still_looking = False
                elif "adjective" in possibilities:
                    best_pos = "A"
                    still_looking = False
                else:
                    # should not happen, pos not recognized
                    best_pos = "!"
                    still_looking = False
            elif len(possibilities) == 0:
                # the morpheme is not recognized
                best_pos = "X"
                still_looking = False
            else:
                best_pos = "?"
                # continue looking
            i += 1
        pos_profile = pos_profile + best_pos
    #print(pos_profile)
    roots = [word for word in message]
    if Lf.prefix:
        root_loc = 1
    else:
        root_loc = 0
    for word_i in range(len(message)):
        if "-" in message[word_i]:
            roots[word_i] = message[word_i].split("-")[root_loc]
    pos_profile = infer(roots, Lf.lang_to_eng, pos_profile, Lf.word_order, Lf.adjective_noun)
    #print(pos_profile, "(inferred)")
    constituents = parse(pos_profile, Lf.word_order, Lf.adjective_noun)
    #print(constituents)
    sentence["subj"]["plural"] = False
    sentence["obj"]["plural"] = False
    sentence["verb"]["past"] = False
    for role in constituents:
        for pos in constituents[role]:
            if "-" in message[constituents[role][pos]]:
                b = message[constituents[role][pos]].split("-")
                if Lf.prefix:
                    c = 0
                else:
                    c = 1
                function = Lf.lang_to_eng["function"][b[c]]
                root = Lf.lang_to_eng[pos][b[1-c]]
                sentence[role][function] = True
                sentence[role][pos] = root
            else:
                b = message[constituents[role][pos]]
                root = Lf.lang_to_eng[pos][b]
                sentence[role][pos] = root
    #print(sentence)

    #if t == "English":
        #return render_eng(sentence)
    #else:
    print("Translation of \"{0}\" from L#{1} to L#{2}:".format(" ".join(message), switch_sign(f), switch_sign(t)))
    tl = render_lang(sentence, Lt.eng_to_lang, Lt.word_order, Lt.adjective_noun, Lt.prefix)
    print("Underlying:", tl)
    print("Surface:", collapse(tl, Lt.morphophonemics, Lt.phonologics))

    # construct the sentence from the word order
    # may need to go through the randomizations so they take seed as parameter

# language class creation

class L:
    def __init__(self, seed):

        # random seed control

        if seed != "E":
            random.seed(seed)
            self.seed = seed

            # phonology construction
        
            self.C_features = gen_features("C", r())
            self.V_features = gen_features("V", r())
            self.all_features = union([self.C_features, self.V_features])
            self.possible_phonemes = []
            for f in self.C_features:
                self.possible_phonemes.extend(phonemes_with_features(C_book, [f]))
            for f in self.V_features:
                self.possible_phonemes.extend(phonemes_with_features(V_book, [f]))
            self.possible_phonemes = remove_duplicates(self.possible_phonemes)
    
            self.categories = {
                "P": intersection([phonemes_with_features(C_book, ["plosive"]), self.possible_phonemes]),
                "F": intersection([phonemes_with_features(C_book, ["fricative"]), self.possible_phonemes]),
                "R": intersection([phonemes_with_features(C_book, ["approximant"]), self.possible_phonemes]),
                "V": intersection([phonemes_with_features(V_book, ["unrounded"]), self.possible_phonemes]),
                "N": intersection([phonemes_with_features(C_book, ["nasal"]), self.possible_phonemes]),
                "J": intersection([phonemes_with_features(C_book, ["affricate"]), self.possible_phonemes])
            }
            for f in sorted(self.categories):
                if f != "V":
                    self.categories[f] = intersection([self.categories[f], self.possible_phonemes])
                    if self.categories[f] == []:
                        del self.categories[f]
            self.categories["C"] = union([self.categories[x] for x in sorted(self.categories) if x != "V"])
        
            # phoneme removals; later, write entropize function
            # note that code in try before exception is thrown will execute
        
            if prob(3,5, r()):
                try:
                    self.categories["C"].remove("ŋ")
                except:
                    pass
            if prob(3,4, r()):
                try:
                    for i in ["pʰ", "tʰ", "tʃʰ", "kʰ"]:
                        self.categories["C"].remove(i)
                except:
                    pass
            if prob(3,5, r()):
                try:
                    self.categories["V"].remove("æ")
                except:
                    pass
            # NOTE: I accidentally removed "ng" from C only, rather than removing
            # it from N before defining C. This actually achieves the desired
            # result of having "ng" finally only. Food for thought.
    
            self.inv = gen_inventory(self.categories, r())
            # DO WE STILL NEED THIS?
            # YES
        
            """
            # just prints the inventory
            for i in sorted(self.inv):
                print(i, self.inv[i])
            """
        
            self.structure = gen_syllable_structure(self.inv, r())
        
            """ # just prints some sample words
            for e in range(5):
                for i in range(7):
                    print(gen_syllable(inv, "CV(CV)(N)"), end=" ")
                print()
            """
    
            # morphology construction

            self.prefix = prob(3, 5, r())

            # morphophonemic rules take place before the removal of
            # morpheme boundaries, as they depend on those
            self.morphophonemics = {
                0: {},
                1: {}
            }

            if self.prefix:
                self.morphophonemics[0] = {"-d":"-r", "-g":"-j", "-b":"-w"}
            else:
                self.morphophonemics[0] = {"r-":"d-", "j-":"g-", "w-":"b-"}

            # phonologic(al) rules take place after the removal of
            # morpheme boundaries and thus can be fed by morphophonemics
            self.phonologics = {
                0:{},
                1:{}
            }
            
            self.phonologics[0] = phonological_rulesets[choose(
                {
                    "long_vowel_diphthongization":1,
                    "long_vowel_monophthongization":2
                },
                r()
            )]
            
            # later, make this randomized and include better interaction
            # between things across morpheme boundaries
    
            # syntax construction
        
            self.word_order = gen_word_order(r())
            self.adjective_noun = prob(1,2, r())
            
            # lexicon construction
    
            # note, from truncnorm documentation:
            # To convert clip values for a specific mean and standard deviation, use:
            # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            self.syllable_number_distribution = {
                1:truncnorm(6,1,3,10,r()),
                2:truncnorm(3,1,0,6,r()),
                3:0#truncnorm(1,1,0,2,r())
            }
    
            self.lang_lex = {}
            for pos in sorted(eng_lex):
                if pos == "function":
                    if self.prefix:
                        struck = parse_structure(self.structure, "initial")
                    else:
                        struck = parse_structure(self.structure, "final")
                    self.lang_lex[pos] = []
                    for word in eng_lex[pos]:
                        syll = collapse(gen_syllable(self.inv, struck, r()), self.phonologics, {0:{}})
                        i = 0
                        while syll in self.lang_lex[pos] and i < len(eng_lex[pos]):
                            syll = collapse(gen_syllable(self.inv, struck, r()), self.phonologics, {0:{}})
                            i += 1
                        if i >= len(eng_lex[pos]):
                            print("Can't escape homophones!")
                            sys.exit()
                        self.lang_lex[pos].append(syll)
                else:
                    self.lang_lex[pos] = []
                    for word in eng_lex[pos]:
                        leng = choose(self.syllable_number_distribution, r())
                        #if prob(9, 10, r()):
                            #leng = 1
                        #else:
                            #leng = 2
                        trans = collapse(gen_word(self.inv, self.structure, leng, r()), self.phonologics, {0:{}})
                        i = 0
                        while trans in self.lang_lex[pos] and i < len(eng_lex[pos]):
                            trans = collapse(gen_word(self.inv, self.structure, leng, r()), self.phonologics, {0:{}})
                            i += 1
                        if i >= len(eng_lex[pos]):
                            print("Can't escape homophones!")
                            sys.exit()
                        self.lang_lex[pos].append(trans)
        
            self.eng_to_lang = {}
            for pos in sorted(eng_lex):
                self.eng_to_lang[pos] = {}
                for i in range(len(eng_lex[pos])):
                    self.eng_to_lang[pos][eng_lex[pos][i]] = self.lang_lex[pos][i]
            self.lang_to_eng = {}
            for pos in sorted(self.lang_lex):
                self.lang_to_eng[pos] = {}
                for i in range(len(self.lang_lex[pos])):
                    self.lang_to_eng[pos][self.lang_lex[pos][i]] = eng_lex[pos][i]
        
            eng_homophones = partial_intersection([eng_lex[pos] for pos in eng_lex], 2)
            self.lang_homophones = partial_intersection([self.lang_lex[pos] for pos in self.lang_lex], 2)
        
            self.eng_to_lang_view = {}
            for pos in sorted(eng_lex):
                for i in range(len(eng_lex[pos])):
                    if True: #if eng_lex[pos][i] in eng_homophones:
                        self.eng_to_lang_view[add_pos(eng_lex[pos][i], pos)] = self.lang_lex[pos][i]
                    #else:
                        #eng_to_lang_view[eng_lex[pos][i]] = lang_lex[pos][i]
            self.lang_to_eng_view = {}
            for pos in sorted(self.lang_lex):
                for i in range(len(self.lang_lex[pos])):
                    if True: #if lang_lex[pos][i] in lang_homophones:
                        self.lang_to_eng_view[add_pos(self.lang_lex[pos][i], pos)] = eng_lex[pos][i]
                    #else:
                        #lang_to_eng_view[lang_lex[pos][i]] = eng_lex[pos][i]
                    
            if len(self.eng_to_lang_view) != len(self.lang_to_eng_view):
                print("Some words were deleted from the language!\nWatch out for homophones!")
                sys.exit()
        else:
            # instantiate English, ignoring most variables because they don't matter
            # since we will not be generating English words or such
    
            self.seed = "E"
            self.word_order = "SVO"
            self.prefix = False
            self.adjective_noun = True
            self.lang_lex = eng_lex
            self.eng_to_lang = {}
            for pos in sorted(eng_lex):
                self.eng_to_lang[pos] = {}
                for i in range(len(eng_lex[pos])):
                    self.eng_to_lang[pos][eng_lex[pos][i]] = eng_lex[pos][i]
            self.lang_to_eng = {}
            for key in self.eng_to_lang:
                self.lang_to_eng[key] = self.eng_to_lang[key]
            self.eng_to_lang["function"] = {"plural": "s", "past": "ed"}            
            self.lang_to_eng["function"] = {"s": "plural", "ed": "past"}
            self.morphophonemics = {}
            self.phonologics = {}

        # execution
    
    def show_typology(self):
        print("Word order:", self.word_order)
        if self.adjective_noun:
            print("Adjectives precede nouns.")
        else:
            print("Adjectives follow nouns.")
        if self.prefix:
            print("This language is predominantly prefixing.")
        else:
            print("This language is predominantly suffixing.")

    def show_lexicon(self):
        for i in range(len(self.eng_to_lang_view)):
            etl = sorted(self.eng_to_lang_view)[i]
            lte = sorted(self.lang_to_eng_view)[i]
            k = 15
            if "(" in etl:
                #etl_red = letter_to_pos[etl[-2]]
                etl_red = etl[:-4]
            else:
                etl_red = etl
            if "(" in lte:
                #lte_red = letter_to_pos[lte[-2]]
                lte_red = lte[:-4]
            else:
                lte_red = lte
            #print(etl, eng_to_lang[letter_to_pos[etl[-2]]][etl_red], "|", lte, lang_to_eng[letter_to_pos[lte[-2]]][lte_red])
            print( # who needs readable print statements
                etl+" "*(k-len(etl)),
                self.eng_to_lang[letter_to_pos[etl[-2]]][etl_red]+
                " "*(k-4-len(self.eng_to_lang[letter_to_pos[etl[-2]]][etl_red])),
                "|", lte+" "*(k-len(lte)),
                self.lang_to_eng[letter_to_pos[lte[-2]]][lte_red]+
                " "*(k-len(self.lang_to_eng[letter_to_pos[lte[-2]]][lte_red])))

    def example_sentences(self):
        print()
        for i in range(5):
            sentence = gen_sentence(eng_lex, r())
            print(render_lang(sentence, E.eng_to_lang, E.word_order, E.adjective_noun, E.prefix))
            print(render_lang(sentence, self.eng_to_lang, self.word_order, self.adjective_noun, self.prefix))
            print(collapse(render_lang(sentence, self.eng_to_lang, self.word_order, self.adjective_noun, self.prefix), self.morphophonemics, self.phonologics))
            print()

E = L("E") # English

# derelict function, no longer used
# do everything (work on moving this stuff into the classes
def do_everything(seed):
    pass
    #random.seed(seed) # the test one was 4565456545654

    # phonology construction

    #C = list_nested_dict(C_book)
    #V = list_nested_dict(V_book)


####### MAIN METHOD #######

if __name__ == "__main__":
    import math
    import random
    import re
    import sys

    #print("What is the task to be performed?\n"
        #"To generate and detail a language, type \"g [seed]\".\n"
        #"To translate, type \"t [from_seed] [to_seed] [message]\".")

    while True:
        task = input()

        if len(task) == 0:
            sys.exit()
        if task[0] == "g":
            s = vet_seed(task[2:], r())
            print("Seed: %d" % switch_sign(s))
            lang = L(s)
            lang.show_typology()
            print()
            lang.show_lexicon()
            lang.example_sentences()
        elif task[0] == "t":
            a = task.split(" ")
            if a[1] != "E":
                from_seed = vet_seed(a[1], r())
            else:
                from_seed = "E"
            #print("FIRST LANGUAGE")
            #from_lang.example_sentences()
            if a[2] != "E":
                to_seed = vet_seed(a[2], r())
            else:
                to_seed = "E"
            #print("SECOND LANGUAGE")
            #to_lang.example_sentences()
            message = a[3:]
            translate(message, from_seed, to_seed)
            print()














