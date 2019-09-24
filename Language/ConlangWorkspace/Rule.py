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
        rules = parse_rule_str(
            s,
            # is_orthographic_rule=is_orthographic_rule,
            add_blanks=add_blanks
        )
        # don't designate unless it will be used, so put the designate() call elsewhere
        return rules

    @staticmethod
    def from_input_and_output_strs(input_str, output_str, is_orthographic_rule=False, grapheme_classes=None):
        if is_orthographic_rule:
            assert grapheme_classes is not None
            # check for hybrid strings
            grapheme_str = input_str
            phoneme_str = output_str
            grapheme_str_is_hybrid = "/" in grapheme_str
            phoneme_str_is_hybrid = "<" in phoneme_str
            if phoneme_str_is_hybrid:
                assert ">" in phoneme_str, "hybrid phoneme str opens grapheme brackets but does not close them: {}".format(phoneme_str)
        else:
            assert grapheme_classes is None
        
        raise NotImplementedError
        
    def to_str(self):
        return self.get_input_str() + " -> " + self.get_output_str()

    def to_notation(self):
        return self.get_input_str() + ">" + self.get_output_str()
        
    def designate(self, s):
        assert type(s) is str, "designation must be str, got {}".format(type(s))
        self.designation = s
        # print("designated {}".format(self))

    def get_segmented_and_replaceability_arrays(self, input_or_output_lst, classes):
        # groups adjacent non-replaceable symbols into one segment
        # e.g. Rule Vs[kh]rVC -> VglVC becomes [V, s[kh]r, V, C] -> [V, gl, V, C]
        assert type(input_or_output_lst) is list, "must be list: {}".format(input_or_output_lst)
        # print("partitioning {}".format(input_or_output_lst))
        segmented_array = []
        replaceability_array = []
        current_unreplaceable_segment = []
        for seg in input_or_output_lst:
            replaceable = seg in classes
            if replaceable:
                if current_unreplaceable_segment != []:
                    segmented_array.append(current_unreplaceable_segment[:])  # don't add object that will later be mutated
                    replaceability_array.append(False)
                    current_unreplaceable_segment = []
                segmented_array.append(seg)
                replaceability_array.append(True)
            else:
                current_unreplaceable_segment.append(seg)

        # at end ,add remaining replaceable stuff if any
        if current_unreplaceable_segment != []:
            segmented_array.append(current_unreplaceable_segment[:])  # don't add object that will later be mutated
            replaceability_array.append(False)
        
        # print("partitioned {} into:\n{}\n{}".format(input_or_output_lst, segmented_array, replaceability_array))
        return segmented_array, replaceability_array

    def partition(self, classes):
        self.input, self.input_replaceability = self.get_segmented_and_replaceability_arrays(self.input, classes)
        self.output, self.output_replaceability = self.get_segmented_and_replaceability_arrays(self.output, classes)
        self.partitioned = True

    def unpartition(self):
        new_inp = []
        new_outp = []
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
        n = len(inp)
        assert n == len(outp), "rule with unequal segmented lengths: {}".format(self)
        assert all(x == y for x, y in zip(self.input_replaceability, self.output_replaceability)), "rule with incompatible replaceabilities: {}".format(self)
        for i in range(n):
            is_replaceable = self.input_replaceability[i]
            if is_replaceable:
                assert type(inp[i]) is type(outp[i]) is str, "{} and {} should be equal and replaceable, but one or both of them is not a string".format(inp[i], outp[i])
                inp_seg = inp[i][0]
                outp_seg = outp[i][0]
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
    
    def __repr__(self):
        rule_str = self.to_str()
        return "Rule #{} : {}".format(self.designation, rule_str)
