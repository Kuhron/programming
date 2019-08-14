import string
import itertools
import random
import ast


class Word:
    def __init__(self, designation, lst):
        self.designation = designation
        self.lst = lst
       
    @staticmethod
    def from_string(s):
        return parse_word_str_to_list(s)
       
class Rule:
    def __init__(self, designation, inp, outp):
        self.designation = designation
        self.input = inp
        self.output = outp
   
    @staticmethod
    def from_str(s):
        return parse_rule_str(s)


def evolve_word(word, rules):
    for rule in rules:
        word = apply_rule(word, rule)
    return word

def expand_classes(rule, classes, used_phonemes):
    inp, outp = rule
    n = len(inp)
    assert n == len(outp), "rule with unequal lengths: {}".format(rule)
    for i in range(n):
        if inp[i] in classes:
            assert outp[i] == inp[i] or outp[i] not in classes
            replace = outp[i] != inp[i]
            res = []
            vals = [x for x in classes[inp[i]] if x in used_phonemes]
            for val in vals:
                replacement = outp[i] if replace else val
                new_inp = inp[:i] + [val] + inp[i+1:]
                new_outp = outp[:i] + [replacement] + outp[i+1:]
                res += expand_classes([new_inp, new_outp], classes, used_phonemes)
            return res
    else:
        return [rule] if inp != outp else []

def parse_word_str_to_list(w):
    lst = []
    inside_brackets = False
    current_item = ""
    for c in w:
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
            elif c == "_":
                # use this to make blanks in rules with string notation
                lst.append("")
            else:
                lst.append(c)
    return lst

def parse_rule_str(inp, classes, used_phonemes):
    rule_strs = inp.split(",")
    all_results = []
    for rule_str in rule_strs:
        rule_inp_str, rule_outp_str = rule_str.split(">")
        rule_inp = parse_word_str_to_list(rule_inp_str)
        rule_outp = parse_word_str_to_list(rule_outp_str)
        if len(rule_inp) != len(rule_outp):
            print("invalid rule given, unequal input and output lengths")
        new_rule = [rule_inp, rule_outp]
        all_results += expand_classes(new_rule, classes, used_phonemes)
       
    return all_results
           
def rule_applies(word, rule):
    word = ["#"] + word + ["#"]
    inp_no_blanks = [x for x in rule[0] if x != ""]
    return list_contains(word, inp_no_blanks)

def get_inputs_that_could_apply(word):
    # ignoring blanks, so check (classless) rules for inclusion in the list that is returned by this function, based on presence of their input without blanks
    # returns triangle number of sublists of word
    if type(word) is str:
        word = parse_word_str_to_list(word)
       
    word = ["#"] + word + ["#"]
    res = []
    for length in range(1, len(word) + 1):
        n_lists = len(word) - length + 1
        for i in range(n_lists):
            res.append(word[i:i+length])
    return res
   
def apply_rule(word, rule):
    if type(word) is str:
        word = parse_word_str_to_list(word)
   
    assert "#" not in word
    inp, outp = rule
    assert inp.count("#") == outp.count("#") <= 2, "too many '#'s in rule {}".format(rule)
   
    word2 = ["#"] + word + ["#"]
    res = sublist_replace(word2,inp, outp)
    res = [x for x in res if x not in ["#", ""]]
    if res == []:
        print("Warning: blocking change that would make {} into a blank word".format(word))
        return word
    if res != word:
        outp_display = "Ø" if outp == "" else "".join(outp)
        print("{} -> {} : {} -> {}".format("".join(inp), outp_display, "".join(word), "".join(res)))
    return res
   
def sublist_replace(lst, old, new):
    assert len(old) == len(new)  # true for this use case
    insert = "" in old
    if insert:
        assert old.count("") == 1
    b = old.index("") if insert else None
    n = len([x for x in old if x != ""])
    m = len([x for x in new if x != ""])
    #print("\n{}, {} -> {}".format(lst, old, new))
    #print("n {}, m {}".format(n, m))
    new_no_blanks = [x for x in new if x != ""]
    word_len = len(lst)
    index_offset = 0
    for i in range(word_len - n + 1):
        j = i + index_offset
        #print(i, index_offset, j)
        slice = lst[j:j+n]
        #print("slice", slice)
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
            #print("insert, slice now =", slice)
        if slice == old:
            lst = lst[:j] + new_no_blanks + lst[j+n:]
            #print("lst now", lst)
            #input("please review")
            index_offset += m-n
    return lst
  
def get_random_rules(n_rules, lexicon, classes):
    res = []
    for _ in range(n_rules):
        raw_w = random.choice(lexicon)
        w = ["#"] + parse_word_str_to_list(raw_w) + ["#"]
        n = len(w)
        max_env_len = min(4, n-1)
        env_len = random.randint(1, max_env_len)
        max_start_index = n - env_len
        min_start_index = 0
        typ = random.choice(
            ["insertion"] * 1 +\
            ["deletion"] * 3 +\
            ["mutation"] * 6
        )
        if env_len == 1 and typ != "insertion":
            # only do insertions if the whole environment is a word boundary
            min_start_index += 1
            max_start_index -= 1
        start_index = random.randint(min_start_index, max_start_index)
        end_index = start_index + env_len
        inp = list(w[start_index : end_index])
        if typ == "insertion":
            # add a blank somewhere
            if inp[0] == "#":
                min_blank_index = 1
            else:
                min_blank_index = 0
            if inp[-1] == "#":
                max_blank_index = len(inp) - 1
            else:
                max_blank_index = len(inp)
            if min_blank_index == 1 and max_blank_index == 0:
                # insertion where input is just ["#"], pick beginning or end of word
                blank_index = random.choice([0, 1])
            else:
                blank_index = random.randint(min_blank_index, max_blank_index)
            inp = inp[:blank_index] + [""] + inp[blank_index:]
            change_index = blank_index
        else:
            while True:
                change_index = random.randrange(len(inp))
                if inp[change_index] != "#":
                    break
       
        # put some classes in inp, with some probability
        # echo these classes in outp, all changes are to nothing or a specific string with no classes
        # e.g., can't make rule like [["#", "", "a"], ["#", "C", "a"]] because don't know which C to insert!
               
        for i, seg in enumerate(inp):
            if len(inp) > 1 and random.random() < 0.3:
                # don't have changes like C -> s or V -> Ø
                classes_with_seg = [c for c in classes if seg in classes[c]]
                if len(classes_with_seg) == 0:
                    continue
               
                c = random.choice(classes_with_seg)
                inp[i] = c
           
        # now copy input as output and do something to it
        outp = inp[:]
        if typ == "insertion":
            c = random.choice(list(classes.keys()))  # make this better later, e.g. don't do C_C -> CfC
            outp[change_index] = random.choice(classes[c])
        elif typ == "deletion":
            outp[change_index] = ""
        elif typ == "mutation":
            if inp[change_index] in classes:
                possibilities = classes[inp[change_index]]
                outp[change_index] = random.choice(possibilities)
            else:
                classes_with_seg = [c for c in classes if inp[change_index] in classes[c]]
                if len(classes_with_seg) == 0:
                    raise Exception("segment to be changed ({}) is not in any class".format(inp[change_index]))
               
                c = random.choice(classes_with_seg)
                outp[change_index] = random.choice([x for x in classes[c] if x != inp[change_index]])
        else:
            raise Exception("unknown change type")
       
        rule = [inp, outp]
        print("generated rule: {} -> {}".format(inp, outp))
        res.append(rule)
   
    return res

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
        assert type(word) is list
        if rule_applies(word, r):
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
        print("".join(original_word))
        word, user_okay_seqs, user_new_rules = user_edit(original_word, classes, used_phonemes)
        #print("got okay seqs", user_okay_seqs)
        #print("got new rules", user_new_rules)
        okay_seqs += user_okay_seqs
        new_rules += user_new_rules
    else:
        word = original_word
    return word, okay_seqs, new_rules
   
def user_edit(word, classes, used_phonemes):
    #print("list form:", word)
    print("word as string:", "".join(word))
    print("input edited word in string form, e.g. *iai or *iai,aai to okay sequence(s), e.g. ViV>VjV,m_a>mba to make rules, or nothing to keep as is")
    okay_seqs = []
    new_rules = []
    while True:
        inp = input("input:\n")
        if inp == "":
            return word, okay_seqs, new_rules
        elif inp[0] == "*":
            okay_seqs += [parse_word_str_to_list(x) for x in inp[1:].split(",")]
        elif ">" in inp:
            rules_from_inp = parse_rule_str(inp, classes, used_phonemes)
            new_rules += rules_from_inp
        else:
            inp = parse_word_str_to_list(inp)
            print("resulting list form:", inp)
            if input("is this correct? (default yes, n for no)") != "n":
                print()
                return inp, okay_seqs, new_rules
       
   
if __name__ == "__main__":
    # use numbers for extra sounds, e.g. n, n1 for palatal, n2 for velar
    # use capital letters for phoneme classes, e.g. C, V, N
   
    classes = {
        "C": ["m", "n", "p", "[ph]", "t", "[th]", "k", "[kh]", "[ts]", "[tsh]", "[tl]", "[tlh]", "s", "h", "l", "r", "j", "w", "b", "d", "g", "[phi]", "[bh]", "x", "ğ", "c", "ć", "č", "ç", "ŕ", "ř", "þ", "đ", "ď", "ņ", "ñ", "š", "f", "v", "z", "ž", "[dź]", "ź", "ś", "[dž]", "ł", "q", "[qh]"],
        "V": ["a", "e", "i", "o", "u", "ù", "ì", "ä", "õ", "ı", "ö", "ü", "ə"],
        "I": ["i", "e"],
        "U": ["u", "o", "a"],
        "[HV]": ["i", "ü", "ı", "u"],
        "Į": ["į", "ų"],
        "F": ["[phi]", "[bh]", "v", "f", "þ", "đ", "s", "z", "ś", "ź", "š", "ž", "x", "ğ", "h", "ł"],
        "T": ["d", "t", "[th]"],
        "N": ["m", "n", "ñ", "ņ"],
        "P": ["p", "[ph]", "b", "t", "[th]", "d", "c", "[ch]", "ģ", "k", "[kh]", "g", "q", "[qh]"],
    }
   
    test_rules = [
        [["V", "C", "V"], ["V", "s", "V"]],
        [["C", "a"], ["z", "a"]],
        [["V", "#"], ["", "#"]],
        [["V", "", "V"], ["V", "j", "u"]],
    ]
    daool_rules = [
        [["V", "h", "V"], ["V", ".", "V"]],
        [["#h"], ["#"]],
        [["C", "", "I"], ["C", "į", "I"]],
        [["C", "", "U"], ["C", "ų", "U"]],
        [["V", "C", "V", "#"], ["V", "C", "", "#"]],
        [["V", "C", "Į", "V", "#"], ["V", "C", "Į", "", "#"]],
        [["į", "I", "V"], ["į", "", "V"]],
        [["ų", "U", "V"], ["ų", "", "V"]],
        [["m", "Į"], ["m", ""]],
        [["nį"], ["n"]],
        [["nų"], ["ņ"]],
        [["[ph]", "Į"], ["f", ""]],
        [["p", "Į"], ["v", ""]],
        [["tį"], ["r"]],
        [["tų"], ["d"]],
        [["[th]į"], ["č"]],
        [["[th]ų"], ["t"]],
        [["[kh]į"], ["s"]],
        [["[kh]ų"], ["x"]],
        [["kį"], ["z"]],
        [["kų"], ["ğ"]],
        [["[tl]", "Į"], ["đ", ""]],
        [["[tlh]į"], ["þ"]],
        [["[tlh]ų"], ["ď"]],
        [["[ts]į"], ["[dź]"]],
        [["[dź]"], ["r"]],
        [["[ts]ų"], ["z"]],
        [["[tsh]į"], ["ć"]],
        [["[tsh]ų"], ["s"]],
        [["sį"], ["š"]],
        [["sų"], ["s"]],
        [["[bh]", "Į"], ["[bh]", ""]],
        [["olį"], ["ùl"]],
        [["ulį"], ["oj"]],
        [["lį"], ["j"]],
        [["lų"], ["l"]],
        [["rį"], ["ŕ"]],
        [["rų"], ["ř"]],
        [["j", "Į"], ["j", ""]],
        [["w", "Į"], ["w", ""]],
        [["V", "h", "Į", "V"], ["V", ".", "", "V"]],
        [["#", "h"], ["#", ""]],
        [["V", "ji"], ["V", "j"]],
        [["C", "ji"], ["C", "je"]],
        [["#", "ji"], ["#", "je"]],
        [["#", "F", "a", "ŕ", "V"], ["#", "F", "", "ŕ", "V"]],
        [["V", "F", "a", "ŕ", "V"], ["V", "F", "", "ŕ", "V"]],
        [["#", "T", "a", "ŕ", "V"], ["#", "T", "", "ŕ", "V"]],
        [["ij", "V"], ["i.", "V"]],
        [["uw", "V"], ["u.", "V"]],
        [["i.i"], ["i"]],
        [["e.e"], ["e"]],
        [["u.u"], ["u"]],
        [["o.o"], ["o"]],
        [["a.a"], ["a"]],
    ]
   
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
   
    test_words = ["pak", "paka", "apak", "apaka", "limiaisa", "tr[ts]kambr", "ağ[dž]oź[dź]iuaruailiłt", "bsgrubs", "aiea", "in", "ni", "m[ts]vrtnelis[ts]qalši"]
    original_words = ["matiali", "nu", "[tlh]ia", "e[tl]aria",
    "[ph]osati", "janio", "weli", "harai",
    "arera", "[tsh]iari", "[tl]uli", "taholi",
    "[tlh]uelima", "ni[tlh]ue[tsh]i", "[bh]ajani", "itianiali",
    "kiuriani", "[kh]iliu", "a[tl]uha", "se[tlh]iura",
    "[th]akariu", "liapapeti", "[kh]a[tlh]i", "luelai",
    "ilai", "[th]i", "eli[th]en", "[th]ai[tsh]a",
    "tarehe", "pareni",
    ]
    verb_roots = ["", "tariaka", "itia", "parenia", "milima"]
    tenses = ["", "ni", "ki"]
    subjs = ["", "ali", "eli", "ari", "atiali", "atieli", "atiari"]
    objs = ["", "api", "epi", "a[tl]i", "upi", "aumi", "umi"]
    original_words += [r+t+s+o for r in verb_roots for t in tenses for o in objs for s in subjs]
    original_words = [x for x in original_words if x != ""]
    targets = ["maraj", "ņu", "þa", "eđŕa",
    "fosar", "jano", "wej", "aři",
    "aŕeř", "ćaŕ", "đoj", "da.ùl",
    "ďejm", "niďeć", "[bh]ajan", "iranaj",
    "zuŕan", "si.u", "ađu.a", "šeþuř",
    "tağŕu", "javaver", "xaþ", "leli",
    "ili", "či", "ejčen", "tis",
    "dŕe", "vŕen",
    #"dŕağ", "dŕağaj", "dŕağej", "dŕağaŕ", "dŕağaraj", "dŕağarej", "dŕağaraŕ", "dŕağejav", "dŕağajev", "dŕağajađ", "dŕağaŕuv", "dŕağajaum", "dŕağajum", "dŕağanaj", "dŕağazaj", "dŕağaņa", "dŕağağa",
    ] + ["TODO"]*1000
   
    # mode = "daool"
    mode = "random"
   
    # words = original_words
    words = test_words
   
    n_steps = len(daool_rules) if mode == "daool" else 20 if mode == "random" else Exception
    words = [parse_word_str_to_list(w) for w in words]
    okay_seqs = []
    for step_i in range(n_steps):
        print("\n---- step", step_i, "----\n")
        #okay_seqs = []  # refresh at every step since language may prefer to change them at some times but not others
        new_rules = []  # simil
        used_phonemes = set()
        for w in words:
            ps = set(w)
            used_phonemes |= ps
           
        if mode == "daool":
            rule = daool_rules[step_i]
        elif mode == "random":
           
            # get a rule
            while True:
                rule = get_random_rules(1, words, classes)[0]
                # print("{} -> {}".format(*rule))
                inp = input("is this a good rule? (default yes, n for no); or, make your own rule, e.g. m_a>mba\n")
                if ">" in inp:
                    # user made their own rule(s)
                    try:
                        expanded_rules = parse_rule_str(inp, classes, used_phonemes)
                        break
                    except AssertionError:
                        print("invalid rule input")
                elif inp != "n":
                    # the given rule was accepted
                    expanded_rules = expand_classes(rule, classes, used_phonemes)
                    break
       
        n_main_rules = len(expanded_rules)
       
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
            word = words[word_i]
            #print("processing word #{}, {}\ntracker: {}".format(word_i, word, rule_catchup_tracker))
            all_rules_this_round = expanded_rules + new_rules
            assert len(all_rules_this_round) == len(rule_catchup_tracker)
            rule_indices_to_apply = [ri for ri in range(len(all_rules_this_round)) if rule_catchup_tracker[ri] < word_i]
            rules_to_apply = [all_rules_this_round[ri] for ri in rule_indices_to_apply]
            inputs_with_effect = get_inputs_that_could_apply(word)
            rules_to_apply_with_effect = rules_to_apply #[r for r in rules_to_apply if [x for x in r[0] if x != ""] in inputs_with_effect]
            #print(word, inputs_with_effect, rules_to_apply_with_effect)
            new_word = evolve_word(word, rules_to_apply_with_effect)
            for ri in rule_indices_to_apply:
                assert rule_catchup_tracker[ri] == word_i - 1, "word_i {}, error in tracker {}".format(word_i, rule_catchup_tracker)  # didn't miss anybody in between
                rule_catchup_tracker[ri] = word_i
            #assert all(x >= word_i - 1 for x in rule_catchup_tracker), "word_i: {}; rule catchup tracker: {}".format(word_i, rule_catchup_tracker)
            # evolve_word should print if a change is made
            assert type(new_word) is list
            new_word, okay_seqs, new_rules = cleanup(new_word, classes, okay_seqs, new_rules, used_phonemes)
            # if get okay seqs, no need to backtrack, just don't ask about them again this step_i
           
            new_words[word_i] = new_word # replace a None if not evolved yet, replace old version if doing rule catchup
           
            # if got some more user rules
            if n_main_rules + len(new_rules) > len(rule_catchup_tracker):
                print("new rules!")
                rule_catchup_tracker += [-1] * (n_main_rules + len(new_rules) - len(rule_catchup_tracker))
                word_i = 0
                continue
            else:
                pass #print("nothing new under the sun")
               
            word_i += 1
            if word_i >= len(words):
                break
               
        # after finish while loop, check that everyone got all the rules
        assert all(x == len(words) - 1 for x in rule_catchup_tracker)
               
        words = new_words
        print("---- finished with step {} ----".format(step_i))
        if any(len(w) < 1 for w in words):
            raise IndexError("blank words created")
           
        print("\n\nlexicon after step", step_i)
        for ow, w, target in zip(original_words, words, targets):
            if ow == w:
                report = "==="
            else:
                report = "-> " + "".join(w)
            print("*{} {} (Daool {})".format("".join(ow), report, target))
           
    if mode == "daool":
        print("\nchecking Daool evolution")
        errors = 0
        for word, target in zip(words, targets):
            if word != target:
                print("result {} != target {}".format("".join(word), target))
                errors += 1
        print("{} errors found".format(errors))
    else:
        print("\nnot checking Daool evolution")
