import string
import itertools
import random
import time
import codecs
import os
import pickle
from copy import deepcopy


def parse_brackets_and_blanks(s):
    assert type(s) is str
    lst = []
    inside_brackets = False
    current_item = ""
    for c in s:
        if inside_brackets:
            assert c != "["
            # no nesting allowed, no meta-digraphs
            if c == "]":
                current_item += c
                lst.append(current_item)
                current_item = ""
                inside_brackets = False
            else:
                current_item += c
        else:
            if c == "[":
                assert current_item == ""
                current_item += c
                inside_brackets = True
            elif c == "-":
                # use this to make blanks in rules with string notation
                lst.append("")
            else:
                lst.append(c)
    return lst

def sublist_replace(lst, old, new):
    print("sublist replace {}, {}, {}".format(lst, old, new))
    assert len(old) == len(new)
    insert = "" in old
    if insert:
        assert old.count("") == 1
    b = old.index("") if insert else None
    n = len([x for x in old if x != ""])
    m = len([x for x in new if x != ""])
    new_no_blanks = [x for x in new if x != ""]
    word_len = len(lst)
    index_offset = 0
    for i in range(word_len - n + 1):
        j = i + index_offset
        slice = lst[j:j+n]
        if insert:
            new_slice = []
            for k in range(n+1):
                if k < b:
                    new_slice.append(slice[k])
                elif k == b:
                    new_slice.append("")
                else:
                    new_slice.append(slice[k-1])
            slice = new_slice
        if slice == old:
            lst = lst[:j] + new_no_blanks + lst[j+n:]
            index_offset += m-n
    return lst

def to_cv(word, classes):
    res = []
    cs = classes["C"]
    vs = classes["V"]
    assert set(cs) & set(vs) == set()
    for c in word:
        res.append("C" if c in cs else "V" if c in vs else "*")
    return res

def list_contains(lst, sub):
    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i+len(sub)] == sub:
            return True
    return False

def cleanup(word, classes, okay_seqs, new_rules, used_phonemes):
    dirty = False
    original_word = word
    for seq in okay_seqs:
        if list_contains(word, seq):
            #print("okayed sequence:", seq)
            word = sublist_replace(word, seq, ["*"] * len(seq))
            
    for r in new_rules:
        # assert type(word) is list
        if r.applies_to_word(word):
            #print("new rule will be applied:", r)
            pass
        
    cv = to_cv(word, classes)
    
    if any(list_contains(word, [x]*3) for c in classes.values() for x in c):
        print("\nword has triple sound")
        dirty = True
        
    if all(x not in word for x in classes["V"]):
        print("\nword has no vowels")
        dirty = True
        
    if any(list_contains(word, [x, y, z]) for x in classes["V"] for y in classes["[HV]"] for z in classes["V"]):
        print("\nword has intervocalic high vowel")
        dirty = True
        
    if list_contains(cv, ["V"]*3):
        print("\nword has 3 vowels in a row")
        dirty = True
        
    if dirty:
        #print(original_word)
        word, user_okay_seqs, user_new_rules = user_edit(original_word, classes, used_phonemes)
        user_okay_seqs_expanded = []
        for seq in user_okay_seqs:
            user_okay_seqs_expanded += Rule.expand_classes(seq, classes, used_phonemes)
        print("got okay seqs", user_okay_seqs_expanded)
        print("got new rules", user_new_rules)
        okay_seqs += user_okay_seqs_expanded
        new_rules += user_new_rules
    else:
        word = original_word
    return word, okay_seqs, new_rules
    
def user_edit(word, classes, used_phonemes):
    #print("list form:", word)
    #print("word as string:", "".join(word))
    print("editing", word)
    print("input edited word in string form, e.g. *iai or *iai,aai to okay sequence(s), e.g. ViV>VjV,m_a>mba to make rules, or nothing to keep as is")
    okay_seqs = []
    new_rules = []
    while True:
        inp = input("input:\n")
        if inp == "":
            return word, okay_seqs, new_rules
        elif inp[0] == "*":
            okay_seqs += [parse_brackets_and_blanks(x) for x in inp[1:].split(",")]
        elif ">" in inp:
            try:
                rules_from_inp = Rule.from_str(inp)
                new_rules += rules_from_inp
            except AssertionError:
                print("invalid rule input")
        else:
            inp_word = Word.from_str(inp)
            print("resulting word:", inp_word)
            if input("is this correct? (default yes, n for no)") != "n":
                print()
                inp_word.designate(word.designation)
                return inp_word, okay_seqs, new_rules

def evolve_user_input_words(rules1, rules2, lexicon):
    new_proto_words = []
    next_designation = 1 + max(int(x.designation) for x in lexicon)
    while True:
        inp = input("\nenter a word to evolve, or nothing to exit\n")
        if inp == "":
            break
        if ">" in inp or "," in inp or "*" in inp:
            # prevent crashing just because I misread the prompt
            print("oops! looks like you didn't give a valid word")
            continue
        inp = Word.from_str(inp)
        inp.designate("None")
        e1 = inp.apply_rules(rules1)
        e2 = inp.apply_rules(rules2)
        print("\nresulting evolutions:\n1. {}\n2. {}\n".format(e1, e2))
        if input("add this proto-word to the lexicon? (y/n, default yes)") != "n":
            inp.designate(str(next_designation))
            next_designation += 1
            # lexicon.append(inp) # put it in original_words instead
            new_proto_words.append(inp)
    return new_proto_words

def create_rules_from_rule_strs(rule_strs, next_rule_designation=0):
    rules = []
    next_rule_designation = 0
    for s in rule_strs:
        rules_from_str = Rule.from_str(s)
        for r in rules_from_str:
            r.designate(str(next_rule_designation))
            next_rule_designation += 1
            rules.append(r)
    return rules, next_rule_designation
    
def build_lexicon_strs(classes):
    assert "C" in classes and "V" in classes
    word_strs = []
    # desig = 0
    used_phonemes = set()
    while True:
        inp = input("make up a new word, enter nothing to generate a word, or enter '*' to stop\ninput: ")
        if inp == "*":
            break
        elif inp == "":
            if word_strs == []:
                print("oops! no words yet to generate from")
                continue
            while True:
                # take an existing word and replace Cs and Vs
                rws = random.choice(word_strs)
                rw = Word.from_str(rws)
                nws = ""
                for ph in rw:
                    if ph in classes["C"]:
                        cands = set(classes["C"])
                    elif ph in classes["V"]:
                        cands = set(classes["V"])
                    else:
                        raise Exception("phoneme {} is not in C or V".format(ph))
                    cands &= used_phonemes
                    nws += random.choice(list(cands))
                if nws not in word_strs:
                    break
            if input("{} : use this word? (y/n, default yes)".format(nws)) == "n":
                continue
            w = Word.from_str(nws)
        else:
            w = Word.from_str(inp)
        # w.designate(desig)
        # desig += 1
        used_phonemes |= w.get_phonemes_used()
        word_strs.append(w.to_str())
        print("current lexicon:", ", ".join(word_strs))
    return word_strs
    
def load_lexicon_from_file(fp):
    with open(fp) as f:
        lines = [x.strip() for x in f.readlines()]
    word_strs = []
    variable_segments = {}
    variable_segment_lines = [x for x in lines if ":" in x]
    lexeme_lines = [x for x in lines if ":" not in x]
    
    for line in variable_segment_lines:
        assert line.count(":") == 1, "bad line " + line
        var_name, val = line.split(":")
        if var_name in variable_segments:
            variable_segments[var_name].append(val)
        else:
            variable_segments[var_name] = [val]
            
    # print(variable_segments)
    for line in lexeme_lines:
        strs_from_line = [""]
        inside_braces = False
        variable_name = ""
        for c in line:
            if c == "{":
                assert not inside_braces
                inside_braces = True
                variable_name = ""
            elif c == "}":
                assert inside_braces
                inside_braces = False
                segment_values = variable_segments[variable_name]
                strs_from_line = [s + val for s in strs_from_line for val in segment_values]
            elif inside_braces:
                variable_name += c
            else:
                strs_from_line = [s + c for s in strs_from_line]
        word_strs += strs_from_line
    return word_strs

def get_random_unicode_character():
    n = random.randint(1, 5000)
    hex_str = str(hex(n))[2:].rjust(4, "0")
    char = eval("\"\\u{0}\"".format(hex_str))
    return char


def get_random_syllable_structure_set():
    possible_onsets = [""] * 1 + ["C"] * 4 + ["CC"] * 0
    possible_nuclei = ["V"]
    possible_codas = [""] * 4 + ["C"] * 1 + ["CC"] * 0 + ["N"] * 0 + ["NC"] * 0
    possible_structures = [onset + nucleus + coda for onset in possible_onsets for nucleus in possible_nuclei for coda in possible_codas]

    result = list(set(random.sample(possible_structures, random.randint(2, len(possible_structures)))))
    input("returning syllable structure set: {}".format(result))
    return result


def get_random_paradigm(inventory, syllable_structure_set):
    input("generating paradigm")
    root = LE2Word.get_random_phone_sequence(random.randint(1, 1), inventory, random.choice(syllable_structure_set))
    print("root:")
    root.print()
    prefixes = [LE2Word([])] + [LE2Word.get_random_phone_sequence(random.randint(1, 1), inventory, random.choice(syllable_structure_set)) for i in range(3)]
    suffixes = [LE2Word([])] + [LE2Word.get_random_phone_sequence(random.randint(1, 1), inventory, random.choice(syllable_structure_set)) for i in range(3)]
    print("prefixes:")
    for w in prefixes:
        w.print()
    print("suffixes:")
    for w in suffixes:
        w.print()
    input()
    return [prefix + root + suffix for prefix in prefixes for suffix in suffixes]


def matches_features_dict(phone, features_dict):
    if features_dict == WordBoundaryPhone().features:
        return phone == WordBoundaryPhone()
    elif type(phone) is not dict:
        return False

    for k, v in features_dict.items():
        if type(v) is list and phone[k] not in v:
            return False
        elif type(v) is not list and phone[k] != v:
            return False
    return True


def get_random_feature_value_from_inventory(inventory):
    phone = random.choice(inventory.phonemes)
    return random.choice([i for i in phone.features.items()])


def get_random_input_language():
    inventory = Inventory.random()
    syllable_structure_set = get_random_syllable_structure_set()
    phonology = Phonology(inventory, syllable_structure_set)
    vocabulary = LE2Lexicon.from_phonology(phonology)
    return (inventory, vocabulary)


def get_input_language_from_file():
    with codecs.open("LanguageEvolution2Input.txt", "rb", "utf-8") as f:
        morphemes = [x.strip() for x in f.readlines()]

    roots = [Word.from_string(x) for x in morphemes if "-" not in x]
    prefixes = [[]] + [Word.from_string(x[:-1]) for x in morphemes if x[-1] == "-"]
    suffixes = [[]] + [Word.from_string(x[1:]) for x in morphemes if x[0] == "-"]
    vocabulary = [list(prefix + root + suffix) for root in roots for prefix in prefixes for suffix in suffixes]

    inventory = []
    for word in (roots + prefixes + suffixes):
        for features_dict in word:
            if features_dict not in inventory:
                inventory.append(features_dict)

    return (inventory, vocabulary)


def generate_language_and_write_to_file():
    # original routine
    inventory, vocabulary = get_random_input_language()
    # inventory, vocabulary = get_input_language_from_file()

    text = [random.choice(vocabulary) for i in range(50)]

    epenthetic_consonant = random.choice([x for x in inventory.phonemes if x.features["syllabicity"] == 0])
    epenthetic_vowel = random.choice([x for x in inventory.phonemes if x.features["syllabicity"] == 3])

    sound_changes = [
        # ({"syllabicity": 0}, "", [WordBoundaryPhone().features], [{"syllabicity": 3}]),  # initial single consonants deleted
        # ({}, {}, {}, {}),  # do nothing
    ] + [SoundChange.get_random_sound_change_from_inventory(inventory) for i in range(30)] + [
        SoundChange({"syllabicity": 0, "voicing": 1}, {"voicing": 0}, PhoneticEnvironment([{"syllabicity": 0, "voicing": 0}], [])),
        SoundChange({"syllabicity": 0, "voicing": 1}, {"voicing": 0}, PhoneticEnvironment([], [{"syllabicity": 0, "voicing": 0}])),  # assimilate clusters to voiceless
        SoundChange("", epenthetic_vowel, PhoneticEnvironment([{"syllabicity": 0}], [{"syllabicity": 0}])),  # epenthetic vowel insertion
        SoundChange("", epenthetic_vowel, PhoneticEnvironment([WordBoundaryPhone().features, {"syllabicity": 0}], [WordBoundaryPhone().features])),  # epenthetic vowel insertion
        SoundChange("", epenthetic_consonant, PhoneticEnvironment([{"syllabicity": 3}], [{"syllabicity": 3}])),  # epenthetic consonant insertion
    ]

    fp = "LanguageEvolution2Output.txt"

    with codecs.open(fp, "wb", "utf-8") as f:
        f.write("inventory:\r\n")
        f.write(inventory.str())
        f.write("\r\n----\r\n")

        f.write("vocabulary:\r\n")
        change_dict = {}
        for word in vocabulary:
            new_word = deepcopy(word)
            for sound_change in sound_changes:
                new_word = sound_change.apply_to_word(new_word)
            ipa = word.get_ipa_str()
            new_ipa = new_word.get_ipa_str()
            change_dict[ipa] = new_ipa
            if new_ipa != ipa:
                f.write("{0} --> {1}\r\n".format(ipa, new_ipa))
            else:
                f.write("{0}\r\n".format(ipa))

        f.write("\r\n---- sample text ----\r\n")
        text_ipas = [word.get_ipa_str() for word in text]
        f.write(" ".join(text_ipas) + "\r\n-->\r\n" + " ".join([change_dict[ipa] for ipa in text_ipas]))
    print("done, written to file {}".format(fp))


def get_syllable_structures_from_user_input_words(words):
    structures = []
    for word in words:
        for syllable in word:
            raise
    # better to convert the inputted words into a canonical format usable by the lexicon


def main():
    classes = DEFAULT_PHONEME_CLASSES
    all_phonemes = set()
    for lst in classes.values():
        all_phonemes |= set(lst)
    
    test_rule_strs = ["k>t", "VCV>VsV", "Ca>za", "V#>_#", "V_V>Vju",
    ]
    daool_rule_strs = [
        "VhV>V.V",
        "#h>#_",
        "C_I>CįI",
        "C_U>CųU",
        "VCV#>VC_#",
        "VCĮV#>VCĮ_#",
        "įIV>į_V",
        "ųUV>ų_V",
        "mĮ>m_",
        "nį>n_",
        "nų>ņ_",
        "[ph]Į>f_",
        "pĮ>v_",
        "tį>r_",
        "tų>d_",
        "[th]į>č_",
        "[th]ų>t_",
        "[kh]į>s_",
        "[kh]ų>x_",
        "kį>z_",
        "kų>ğ_",
        "[tl]Į>đ_",
        "[tlh]į>þ_",
        "[tlh]ų>ď_",
        "[ts]į>[dź]_",
        "[dź]>r",
        "[ts]ų>z_",
        "[tsh]į>ć_",
        "[tsh]ų>s_",
        "sį>š_",
        "sų>s_",
        "[bh]Į>[bh]_",
        "olį>ùl_",
        "ulį>oj_",
        "lį>j_",
        "lų>l_",
        "rį>ŕ_",
        "rų>ř_",
        "jĮ>j_",
        "wĮ>w_",
        "VhĮV>V._V",
        # "#h>#_",  # already done
        "Vji>Vj_",
        "Cji>Cje",
        "#ji>#je",
        "#FaŕV>#F_ŕV",
        "VFaŕV>VF_ŕV",
        "#TaŕV>#T_ŕV",
        "ijC>i_C",
        "ijV>i.V",
        "uwC>u_C",
        "uwV>u.V",
        "i.i>i__",
        "e.e>e__",
        "u.u>u__",
        "o.o>o__",
        "a.a>a__",
    ]
    daool_rules, next_daool_rule_designation = create_rules_from_rule_strs(daool_rule_strs)
    expanded_daool_rules = []
    for r in daool_rules:
        expanded_daool_rules += r.get_specific_cases(classes, all_phonemes)
    
    # android keyboard orthography for modern Daool
    # nn ņ
    # t d, tt t
    # d ď (one char, háček)
    # v bh
    # f v, ff f
    # tl đ, ttl þ
    # s z, ss s
    # zl š
    # z ć
    # tzl č
    # x ğ, xx x
    # h .
    # rl ŕ, w ř
    # ll i/j, u u/w
    # ol ùl
    # ul oj
    
    test_word_strs = ["pak", "paka", "apak", "apaka", "limiaisa", "tr[ts]kambr", "ağ[dž]oź[dź]iuaruailiłt", "bsgrubs", "aiea", "in", "ni", "m[ts]vrtnelis[ts]qalši"]
    proto_daellic_word_strs = ["matiali", "nu", "[tlh]ia", "e[tl]aria", 
    "[ph]osati", "janio", "weli", "harai", 
    "arera", "[tsh]iari", "[tl]uli", "taholi",
    "[tlh]uelima", "ni[tlh]ue[tsh]i", "[bh]ajani", "ilisiani",
    "kiuriani", "[kh]iliu", "a[tl]uha", "se[tlh]iura",
    "[th]akariu", "liapapeti", "[kh]a[tlh]i", "luelai",
    "ilai", "[th]i", "eli[th]en", "[th]ai[tsh]a",
    "tarehe",
    ]
    verb_roots = ["tariak", "iti", "pareni", "milim"]
    #tenses = ["a", "ani", "aki"]
    subjs = ["ali", "eli", "ari", "atiali", "atieli", "atiari"]
    verb_suffixes = ["a"] + subjs
    verb_suffixes += ["anaha"] + ["ani" + x for x in subjs]
    verb_suffixes += ["akaha"] + ["aki" + x for x in subjs]
    #objs = [""]#, "api", "epi", "a[tl]i", "upi", "aumi", "umi"]
    #proto_daellic_word_strs += [r+t+s+o for r in verb_roots for t in tenses for o in objs for s in subjs]
    proto_daellic_word_strs += [r+s for r in verb_roots for s in verb_suffixes]
    #proto_daellic_word_strs = [x for x in proto_daellic_word_strs if x != ""]
    targets = ["maraj", "ņu", "þa", "eđŕa", 
    "fosar", "jano", "wej", "aři", 
    "aŕeř", "ćaŕ", "đoj", "da.ùl", 
    "ďejm", "niďeć", "[bh]ajan", "išan",
    "zuŕan", "si.u", "ađu.a", "šeþuř",
    "tağŕu", "javaver", "xaþ", "leli",
    "ili", "či", "ejčen", "tis",
    "dŕe",
    ]
    verb_roots = ["dŕağ", "ir", "vŕen", "mim"]
    #tenses = ["", "an", "az"]
    subjs = ["aj", "ej", "aŕ", "araj", "arej", "araŕ"]
    verb_suffixes = [""] + subjs
    verb_suffixes += ["aņa"] + ["an" + x for x in subjs]
    verb_suffixes += ["ağa"] + ["az" + x for x in subjs]
    #objs = [""]#, "av", "ev", "ađ", "uv", "aum", "um"]
    #targets += [r+t+s+o for r in verb_roots for t in tenses for o in objs for s in subjs]
    targets += [r+s for r in verb_roots for s in verb_suffixes]
    #targets = [x for x in targets if x != ""]
    targets += ["TODO"] * 1000 + ["END OF TARGETS"]
    
    # mode = "daool"
    # mode = "random_daellic"
    # mode = "test"
    mode = "random"
    
    def n_step_generator(n):
        for i in range(n):
            yield i
            
    def infinite_step_generator():
        i = 0
        while True:
            yield i
            i += 1
    
    if mode == "daool":
        #n_steps = len(daool_rules)
        step_generator = n_step_generator(len(daool_rules))
        word_strs = proto_daellic_word_strs
        rule_strs = daool_rule_strs
    elif mode == "test":
        #n_steps = 5
        step_generator = n_step_generator(5)
        word_strs = test_word_strs
        rule_strs = test_rule_strs
    elif mode == "random":
        #n_steps = int(input("number of steps to evolve?: "))
        step_generator = infinite_step_generator()
        if input("build lexicon from scratch? (y/n, default yes)") != "n":
            word_strs = build_lexicon_strs(classes)
        else:
            # lexicon_file = "test_lexicon.txt"
            lexicon_file = "simple_lexicon.txt"
            word_strs = load_lexicon_from_file(lexicon_file)
            print("loaded lexicon:")
            for w in word_strs:
                print("*"+w)
        rule_strs = []
    elif mode == "random_daellic":
        #n_steps = int(input("number of steps to evolve?: "))
        step_generator = infinite_step_generator()
        word_strs = proto_daellic_word_strs
        rule_strs = []
    else:
        raise Exception("invalid mode " + mode)
    
    words = [Word.from_str(s, designation=str(desig)) for desig, s in enumerate(word_strs)]
    original_words = words[:]
    next_word_designation = len(words)
    
    rules, next_rule_designation = create_rules_from_rule_strs(rule_strs, next_rule_designation=0)
    
    okay_seqs = []
    rule_history = []  # only for expanded rules
    for step_i in step_generator:
        print("\n---- step", step_i, "----\n")
        new_rules = []  # reset at every step
        used_phonemes = set()
        for w in words:
            ps = w.get_phonemes_used()
            used_phonemes |= ps
            
        try:
            rule = rules[step_i]
            print("using pre-defined rule {}".format(rule))
            expanded_rules = rule.get_specific_cases(classes, used_phonemes)
        except IndexError:
            # no pre-defined rule, so get a rule at random or from user
            print("no pre-defined rule at this step; generating random rule")
            while True:
                rule = Rule.get_random_rules(1, words, classes)[0]
                # print("{} -> {}".format(*rule))
                inp = input("is this a good rule? (default yes, n for no); or, make your own rule, e.g. m_a>mba\n")
                if ">" in inp:
                    # user made their own rule(s)
                    try:
                        rules_from_str = Rule.from_str(inp)
                        expanded_rules = []
                        for ri, rule in enumerate(rules_from_str):
                            rule.designate(str(next_rule_designation))
                            next_rule_designation += 1
                            expanded_rules += rule.get_specific_cases(classes, used_phonemes)
                            print("expanded_rules now has len", len(expanded_rules))
                        print("got user rules in new-step phase:", rules_from_str)

                        # ensure that used phonemes will contain anything new created by the rule
                        for r in expanded_rules:
                            used_phonemes |= set(r.get_output_phonemes_used())
                        break
                    except AssertionError:
                        print("invalid rule input")
                elif inp != "n":
                    # the given rule was accepted
                    # give it a designation
                    rule.designate(str(next_rule_designation))
                    next_rule_designation += 1
                    expanded_rules = rule.get_specific_cases(classes, used_phonemes)
                    break
                else:
                    # user rejected rule
                    continue
                    
        n_main_rules = len(expanded_rules)
        
        all_rules_this_round = expanded_rules
        rule_catchup_tracker = [-1 for r in expanded_rules]  
        # if get new rule from user, add a term here and start at beginning of list of words, applying all user rules that have not been applied yet
        # e.g. [5] indicating word_i 5 has had the main rule applied, user makes a new rule
        # then [5, -1], then start at word_i 0 again
        # suppose at word_i 2 on this second sweep, user makes another rule
        # then [5, 2, -1], then user makes no more rules this round
        # for word_i, if array element is less than word_i then rule has not been applied, so apply it
        
        word_i = 0
        new_words = [None] * len(words)
        while True:
            #print("sleepy time")
            #time.sleep(1)
            word = words[word_i]
            #print("processing word #{}, {}\nrules: {}\ntracker: {}".format(word_i, word, all_rules_this_round, rule_catchup_tracker))
            
            new_rules = []  # clear it so things won't be overwritten if user makes yet another rule this step (i.e. there will be a third "start" (word_i=0) to this while loop)
            assert len(all_rules_this_round) == len(rule_catchup_tracker), "{} rules but tracker has {}".format(len(all_rules_this_round), len(rule_catchup_tracker))
            rule_indices_to_apply = [ri for ri in range(len(all_rules_this_round)) if rule_catchup_tracker[ri] < word_i]
            rules_to_apply = [all_rules_this_round[ri] for ri in rule_indices_to_apply]
            #inputs_with_effect = word.get_inputs_that_could_apply()
            rules_to_apply_with_effect = rules_to_apply #[r for r in rules_to_apply if [x for x in r[0] if x != ""] in inputs_with_effect]
            #print(word, inputs_with_effect, rules_to_apply_with_effect)
            new_word = evolve_word(word, rules_to_apply_with_effect)
            for ri in rule_indices_to_apply:
                assert rule_catchup_tracker[ri] == word_i - 1, "word_i {}, error in tracker {}".format(word_i, rule_catchup_tracker)  # didn't miss anybody in between
                rule_catchup_tracker[ri] = word_i
            #assert all(x >= word_i - 1 for x in rule_catchup_tracker), "word_i: {}; rule catchup tracker: {}".format(word_i, rule_catchup_tracker)
            # evolve_word should print if a change is made
            #print("word is now",new_word)
            words[word_i] = new_word  # update before cleanup so it will be saved with whatever change just happened
            if mode != "daool":
                new_word, okay_seqs, new_rules = cleanup(new_word, classes, okay_seqs, new_rules, used_phonemes)
                #print("cleanup results:\n{}\n{}\n{}".format(new_word, okay_seqs, new_rules))
                #input("continue")
                # if get okay seqs, no need to backtrack, just don't ask about them again this step_i
            
                words[word_i] = new_word # replace a None if not evolved yet, replace old version if doing rule catchup
            
            # if got some more user rules
            expanded_new_rules = []
            for r in new_rules:
                r.designate(str(next_rule_designation))
                next_rule_designation += 1
                expanded_new_rules += r.get_specific_cases(classes, used_phonemes)
            
            for r in expanded_new_rules:
                print("appending to rules this round:", r)
                all_rules_this_round.append(r)
                rule_catchup_tracker.append(-1)
                
            # new_rules = expanded_new_rules
            # if n_main_rules + len(new_rules) > len(rule_catchup_tracker):
                # print("new rules!")
                # print(new_rules)
                # rule_catchup_tracker += [-1] * (n_main_rules + len(new_rules) - len(rule_catchup_tracker))
            if len(expanded_new_rules) > 0:
                # go back to first word and apply these new rule
                print("resetting to word_i 0 to apply new rules")
                word_i = 0
                continue
            else:
                pass #print("nothing new under the sun")
                
            word_i += 1
            if word_i >= len(words):
                break
                
        # after finish while loop, check that everyone got all the rules
        assert all(x == len(words) - 1 for x in rule_catchup_tracker)
        rule_history += all_rules_this_round[:]
                
        #words = new_words
        print("---- finished with step {} ----".format(step_i))
        if any(len(w) < 1 for w in words):
            raise IndexError("blank words created")
            
        print("\n\nlexicon after step", step_i)
        for ow, w, target in zip(original_words, words, targets):
            report = "*{}".format(ow.to_str())
            if ow == w:
                report += " ==="
            else:
                report += " -> " + w.to_str()
            if mode in ["daool", "random_daellic"]:
                report += " (Daool {})".format(target)
            print(report)
            
        if mode in ["random", "random_daellic"]:
            new_proto_words = evolve_user_input_words(rule_history, expanded_daool_rules, words)
            original_words += new_proto_words
            words += [evolve_word(w, rule_history) for w in new_proto_words]
        if mode != "daool" and input("press enter to continue, or enter '*' to stop: ") == "*":
            break
            
    if mode == "daool":
        print("\nchecking Daool evolution")
        errors = 0
        for word, target in zip(words, targets):
            target = Word.from_str(target)
            if word != target:
                print("result {} != target {}".format("".join(word), target))
                errors += 1
        print("{} errors found".format(errors))
    else:
        print("\nnot checking Daool evolution")
        
    print("rule history")
    for r in rule_history:
        print(r)


if __name__ == "__main__":    
    main()