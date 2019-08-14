import string
import itertools
import random
import ast

def evolve_word(word, rules):
    for rule in rules:
        word = apply_rule(word, rule)
    return word

def expand_classes(rule, classes):
    inp, outp = rule
    n = len(inp)
    assert n == len(outp), "rule with unequal lengths: {}".format(rule)
    for i in range(n):
        if inp[i] in classes:
            assert outp[i] == inp[i] or outp[i] not in classes
            replace = outp[i] != inp[i]
            res = []
            for val in classes[inp[i]]:
                replacement = outp[i] if replace else val
                new_inp = inp[:i] + [val] + inp[i+1:]
                new_outp = outp[:i] + [replacement] + outp[i+1:]
                res += expand_classes([new_inp, new_outp], classes)
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
   
def apply_rule(word, rule):
    # print("-- rule no classes")
    # print(rule)
    if type(word) is str:
        word = parse_word_str_to_list(word)
   
    assert "#" not in word
    #list_inp = rule[0]
    #list_outp = rule[1]
    #inp = "".join(list_inp)
    inp, outp = rule
    #outp = "".join(list_outp)
    assert inp.count("#") == outp.count("#") < 2, "too many '#'s in rule {}".format(rule)
    if "#" in inp:
        if inp[0] == "#":
            # only change the front of the word
            word2 = ["#"] + word
        elif inp[-1] == "#":
            # back
            word2 = word + ["#"]
        else:
            raise Exception("rule has '#' in the middle: {}".format(rule))
    else:
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
        res.append("C" if c in cs else "V" if c in vs else Exception)
    return res

def list_contains(lst, sub):
    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i+len(sub)] == sub:
            return True
    return False

def cleanup(word, classes):
    dirty = False
    cv = to_cv(word, classes)
    if any(list_contains(word, [x]*3) for c in classes.values() for x in c):
        print("triple letter found:", word)
        dirty = True
    if all(x not in word for x in classes["V"]):
        print("word has no vowels:", word)
        dirty = True
    if any(list_contains(word, [x, y, z]) for x in classes["V"] for y in classes["[HV]"] for z in classes["V"]):
        print("word has intervocalic high vowel:", word)
        dirty = True
    if list_contains(cv, ["V"]*3):
        print("word has 3 vowels in a row:", word)
        dirty = True
       
    if dirty:
        word, okay_seqs, new_rules = user_edit(word)
    return word, [], []
   
def user_edit(word):
    print("list form:", word)
    print("string form:", "".join(word))
    print("input edited word in string form, e.g. *iai or *iai,aai to okay sequence(s), e.g. ViV>VjV,m_a>mba to make rules, or nothing to keep as is")
    while True:
        inp = input()
        okay_seqs = []
        new_rules = []
        # TODO: user can input okay sequence such as "iau" so it will not be asked about again, or a rule such as "VuV" -> "VwV" to decrease tedium
        if inp == "":
            return word, [], []
        elif inp[0] == "*":
            okay_seqs = (inp[1:]).split(",")
        elif ">" in inp:
            rule_strs = inp.split(",")
            for rule_str in rule_strs:
                rule_inp_str, rule_outp_str = rule_str.split(">")
                rule_inp = parse_word_str_to_list(rule_inp_str)
                rule_outp = parse_word_str_to_list(rule_outp_str)
                new_rule = [rule_inp, rule_outp]
                new_rules.append(new_rule)
        else:
            inp = parse_word_str_to_list(inp)
            print("resulting list form:", inp)
            if input("is this correct? (default yes, n for no)") != "n":
                print()
                return new_word, okay_seqs, new_rules
       
   
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
   
    original_words = ["matiali", "nu", "[tlh]ia", "e[tl]aria",
    "[ph]osati", "janio", "weli", "harai",
    "arera", "[tsh]iari", "[tl]uli", "taholi",
    "[tlh]uelima", "ni[tlh]ue[tsh]i", "[bh]ajani", "itianiali",
    "kiuriani", "[kh]iliu", "a[tl]uha", "se[tlh]iura",
    "[th]akariu", "liapapeti", "[kh]a[tlh]i", "luelai",
    "ilai", "[th]i", "eli[th]en", "[th]ai[tsh]a",
    "tarehe", "pareni",
    ]
    verb_roots = ["tariaka", "itia", "parenia", "milima"]
    tenses = ["", "ni", "ki"]
    subjs = ["", "ali", "eli", "ari", "atiali", "atieli", "atiari"]
    objs = ["", "api", "epi", "a[tl]i", "upi", "aumi", "umi"]
    original_words += [r+t+s+o for r in verb_roots for t in tenses for s in subjs for o in objs]
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
   
    n_steps = len(daool_rules) if mode == "daool" else 20 if mode == "random" else Exception
    words = original_words[:]
    for step_i in range(n_steps):
        if mode == "daool":
            rule = daool_rules[step_i]
        elif mode == "random":
            while True:
                rule = get_random_rules(1, words, classes)[0]
                # print("{} -> {}".format(*rule))
                if input("is this a good rule? (default yes, n for no)") != "n":
                    break
       
        new_words = []
        expanded_rules = expand_classes(rule, classes)
   
        for word in words:
            new_word = evolve_word(word, expanded_rules)
            # evolve_word should print if a change is made
            new_word = cleanup(new_word, classes)
            new_words.append(new_word)
        words = new_words
        print("----")
        if any(len(w) < 1 for w in words):
            raise IndexError("blank words created")
   
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
