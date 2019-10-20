from LanguageEvolutionTools import parse_brackets_and_blanks
from SegmentSequence import SegmentSequence

import random


class Rule:
    def __init__(self, inp, outp, designation=None):
        self.designation = designation
        self.input = inp
        self.output = outp
        self.input_replaceability = None
        self.output_replaceability = None
        self.partitioned = False
    
    @staticmethod
    def from_str(s, add_blanks=True):
        rule_strs = s.split(",")
        all_results = []
        for rule_str in rule_strs:
            if rule_str.count(">") != 1:
                print("skipping invalid rule_str:", rule_str)
                continue
            rule_inp_str, rule_outp_str = rule_str.split(">")
            if rule_inp_str.count("-") > 1:
                print("only insertions with one blank are accepted right now; please split this into a series of rules:", rule_str)
                continue
            rule_inp = parse_brackets_and_blanks(rule_inp_str)
            rule_outp = parse_brackets_and_blanks(rule_outp_str)
            if add_blanks:
                if len(rule_inp) != len(rule_outp):
                    lri = len(rule_inp)
                    lro = len(rule_outp)
                    input_shorter = lri < lro
                    shorter_one, shorter_len, longer_len = (rule_inp, lri, lro) if input_shorter else (rule_outp, lro, lri)
                    shorter_one += [""] * (longer_len - shorter_len)
                    if input_shorter:
                        rule_inp = shorter_one
                    else:
                        rule_outp = shorter_one
                if len(rule_inp) != len(rule_outp):
                    raise AssertionError("invalid rule given, unequal input and output lengths\ninput: {}\noutput: {}".format(rule_inp, rule_outp))
            new_rule = Rule(rule_inp, rule_outp)
            #all_results += new_rule.get_specific_cases(classes, used_phonemes)  # do expansion later
            all_results.append(new_rule)

        # don't designate unless it will be used, so put the designate() call elsewhere    
        return all_results

    @staticmethod
    def from_input_and_output_strs(input_str, output_str):
        input_seq = SegmentSequence.from_str(input_str)
        output_seq = SegmentSequence.from_str(output_str)
        rule = Rule(input_seq, output_seq)
        return rule
        
    def to_str(self):
        return self.get_input_str() + " -> " + self.get_output_str()

    def to_notation(self):
        return self.get_input_str() + ">" + self.get_output_str()
        
    def designate(self, s):
        assert type(s) is str, "designation must be str, got {}".format(type(s))
        self.designation = s

    @staticmethod
    def get_replaceability_array(lst, classes):
        res = []
        for seg in lst:
            if type(seg) is list:
                # unhashable type
                res.append(False)
            else:
                res.append(seg in classes)
        return res

    @staticmethod
    def partition_input_and_output_lists(input_lst, output_lst, classes):
        input_replaceability = Rule.get_replaceability_array(input_lst, classes)
        output_replaceability = Rule.get_replaceability_array(output_lst, classes)
        # check compatibility
        for inp_r, outp_r in zip(input_replaceability, output_replaceability):
            # reject an unreplaceable (e.g. /k/) turning into a replaceable (e.g. /C/)
            assert not (outp_r and not inp_r), "incompatible replaceabilities: {} > {}".format(input_lst, output_lst)
        # now make an array that they will both be partitioned according to
        # want to "or" the boolean arrays
        # but since you will only ever have T>T, F>F, or T>F, the or should be the same as the input
        total_replaceability = [a or b for a, b in zip(input_replaceability, output_replaceability)]
        assert total_replaceability == input_replaceability, "logic error in replaceability arrays {} > {}".format(input_replaceability, output_replaceability)
        partitioned_input = Rule.partition_list(input_lst, replaceability=total_replaceability)
        partitioned_output = Rule.partition_list(output_lst, replaceability=total_replaceability)

        # disallow changing things into or from word boundaries
        for a, b in zip(partitioned_input, partitioned_output):
            assert not (a == "" and b == ""), "Changing to or from word boundary is not allowed: {} > {}".format(input_lst, output_lst)
        return partitioned_input, partitioned_output

    @staticmethod
    def partition_list(lst, replaceability=None, classes=None):
        if replaceability is None:
            assert classes is None, "either replaceability or classes must be supplied"
            replaceability = Rule.get_replaceability_array(lst, classes)
        # put sequences of unreplaceables together
        partitioned = []
        current_unreplaceable_segment = []
        for x, r in zip(lst, replaceability):
            if r:
                if current_unreplaceable_segment != []:
                    partitioned.append(current_unreplaceable_segment[:])  # don't add object that will later be mutated
                    current_unreplaceable_segment = []
                partitioned.append(x)
            else:
                current_unreplaceable_segment.append(x)

        # at end ,add remaining replaceable stuff if any
        if current_unreplaceable_segment != []:
            partitioned.append(current_unreplaceable_segment[:])  # don't add object that will later be mutated
        
        return partitioned

    def partition(self, classes):
        self.input, self.output = Rule.partition_input_and_output_lists(self.input, self.output, classes)
        # save the replaceability attributes AFTER partitioning so the length will match
        self.input_replaceability = Rule.get_replaceability_array(self.input, classes)
        self.output_replaceability = Rule.get_replaceability_array(self.output, classes)
        self.partitioned = True

    def unpartition(self):
        new_inp = Rule.flatten_partitioned_list(self.input)
        new_outp = Rule.flatten_partitioned_list(self.output)
        self.input = new_inp
        self.output = new_outp
        self.input_replaceability = None
        self.output_replaceability = None
        self.partitioned = False

    @staticmethod
    def flatten_partitioned_list(lst):
        res = []
        for x in lst:
            if type(x) is list:
                for y in x:
                    res.append(y)
            else:
                res.append(x)
        return res
        
    def get_specific_cases(self, classes, used_phonemes=None):
        if not self.partitioned:
            self.partition(classes)
        inp = self.input
        outp = self.output
        # print("getting specific cases of {}".format(self))
        # print(inp, outp)
        n = len(inp)
        assert n == len(outp)
        for i in range(n):
            is_replaceable = self.input_replaceability[i]
            if is_replaceable:
                assert type(inp[i]) is type(outp[i]) is str, "{} and {} should be equal and replaceable, but one or both of them is not a string".format(inp[i], outp[i])
                inp_seg = inp[i]
                outp_seg = outp[i]
                assert outp_seg == inp_seg or outp_seg not in classes
                replace = outp_seg != inp_seg
                res = []
                if used_phonemes is None:
                    # default behavior is to get all possible expansions given classes
                    vals = [x for x in classes[inp_seg]]
                else:
                    vals = [x for x in classes[inp_seg] if x in used_phonemes]
                
                for val_i, val in enumerate(vals):
                    replacement = outp_seg if replace else val
                    new_inp = inp[:i] + [val] + inp[i+1:]
                    new_outp = outp[:i] + [replacement] + outp[i+1:]
                    designation = "{}.{}".format(self.designation, val_i)
                    new_rule = Rule(new_inp, new_outp)
                    new_rule.designate(designation)
                    new_rule.unpartition()
                    res += new_rule.get_specific_cases(classes, used_phonemes)
                return res
        else:
            return [self]  # may be a rule that does not change anything, but this is needed for orthography
            
    @staticmethod
    def expand_classes(s, classes, used_phonemes):
        n = len(s)
        for i in range(n):
            if s[i] in classes:
                #replace = outp[i] != inp[i]
                res = []
                vals = [x for x in classes[s[i]] if x in used_phonemes]
                for val in vals:
                    new_s = s[:i] + [val] + s[i+1:]
                    res += Rule.expand_classes(new_s, classes, used_phonemes)
                return res
        else:
            return [s]
            
    def get_output_phonemes_used(self):
        if self.has_classes():
            raise Exception("can only get output phonemes from specific case, but this rule is a general case: {}".format(self))
        return set(self.output)
        
    def get_input_str(self):
        if self.partitioned:
            input_lst = Rule.flatten_partitioned_list(self.input)
        else:
            input_lst = self.input
        return "".join(("-" if x == "" else x) for x in input_lst)
        
    def get_output_str(self):
        if self.partitioned:
            output_lst = Rule.flatten_partitioned_list(self.output)
        else:
            output_lst = self.output
        s = "".join(("-" if x == "" else x) for x in output_lst)
        return s
        
    def has_classes(self):
        return any(x in string.ascii_uppercase for x in self.to_str())
        
    def get_input_no_blanks(self):
        return [x for x in self.input if x != ""]

    def applies_to_word(self, word):
        word = word.with_word_boundaries()
        inp_no_blanks = self.get_input_no_blanks()
        return list_contains(word, inp_no_blanks)
    
    def __repr__(self):
        rule_str = self.to_str()
        return "Rule #{} : {}".format(self.designation, rule_str)

    @staticmethod
    def get_random_rules(n_rules, lexicon, classes):
        res = []
        for _ in range(n_rules):
            w = random.choice(lexicon)
            w = w.with_word_boundaries()
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
            assert min_start_index <= max_start_index, "non-overlapping indices with parameters\nword = {w}\ntyp = {typ}\nenv_len = {env_len}\nindices = {min_start_index}, {max_start_index}".format(**locals())
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
                outp[change_index] = random.choice(tuple(classes[c]))
            elif typ == "deletion":
                outp[change_index] = ""
            elif typ == "mutation":
                if inp[change_index] in classes:
                    possibilities = classes[inp[change_index]]
                    outp[change_index] = random.choice(tuple(possibilities))
                else:
                    classes_with_seg = [c for c in classes if inp[change_index] in classes[c]]
                    if len(classes_with_seg) == 0:
                        raise Exception("segment to be changed ({}) is not in any class".format(inp[change_index]))
                    
                    c = random.choice(classes_with_seg)
                    outp[change_index] = random.choice([x for x in classes[c] if x != inp[change_index]])
            else:
                raise Exception("unknown change type")
            
            rule = Rule(inp, outp)  # don't designate it until it is accepted for use
            # print("generated rule: {}".format(rule))
            res.append(rule)
        
        return res
