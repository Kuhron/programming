from Grapheme import Grapheme
from InflectionForm import InflectionForm
from Lexeme import Lexeme
from Morpheme import Morpheme
from Phone import Phone
from Phoneme import Phoneme
from Rule import Rule
from Word import Word
from OrthographyConverter import OrthographyConverter

import docx
from docx.shared import Pt


class CommandProcessor:
    def __init__(self, gui, orthography_converter):
        self.command_history = []
        self.time_history = []
        self.gui = gui
        self.orthography_converter = orthography_converter

        # language-related variables for the processor to keep track of
        # maybe should move these to be attributes of the language, as long as processor can see them somehow
        self.full_inflections_by_part_of_speech = {}
        self.affixes_by_part_of_speech_and_feature = {}
        self.inflection_hierarchy_by_part_of_speech = {}
        self.inflection_templates_by_part_of_speech = {}

    @staticmethod
    def is_command_entry(s):
        return s[0] != "#"
    
    def process_command(self, command_str):
        command_str = command_str.strip()

        if command_str == "" or command_str[0] == "#":
            pass
        else:
            self.process_command_entry(command_str)

        # if command was processed successfully
        self.command_history.append(command_str)
        self.gui.update_command_history_display()

    def process_command_entry(self, ce):
        command, *rest = ce.split(" ")
        print("got command {} with args {}".format(command, rest))

        command_dict = {
            "lex": self.process_lexeme_command_entry,
            "include": self.process_include_command_entry,
            "phone": self.process_phone_command_entry,
            "phoneme": self.process_phoneme_command_entry,
            "allophone": self.process_allophone_command_entry,
            "pos": self.process_pos_command_entry,
            "inflect": self.process_inflect_command_entry,
            "sc": self.process_sound_change_command_entry,
            "graph": self.process_graph_command_entry,
            "ortho": self.process_ortho_command_entry,
            "write": self.process_write_command_entry,
            "read": self.process_read_command_entry,
            "time": self.process_time_command_entry,
        }

        if command in command_dict:
            func = command_dict[command]
            func(rest)
        else:
            raise NameError("unknown command \'{}\'".format(command))

    def process_include_command_entry(self, args):
        # e.g. include DefaultPhonology.cwg
        fp, = args
        self.load_input_file(fp)

    def process_phone_command_entry(self, args):
        # e.g. phone m place=bilabial manner=nasal voicing=1
        phone_symbol, *features = args
        features_dict = {}
        for feature_str in features:
            key, value = feature_str.split("=")
            assert key not in features_dict
            features_dict[key] = value
        phone = Phone(phone_symbol, features_dict)
        self.gui.language.add_phone(phone)
        self.gui.update_phonology_displays()

    def process_phoneme_command_entry(self, args):
        # e.g. phoneme m
        phoneme_symbol, classes_str = args
        classes = classes_str.split(",")
        classes = ["/"+c+"/" for c in classes]
        phoneme = Phoneme(phoneme_symbol)
        self.gui.language.add_phoneme(phoneme, classes)
        self.gui.update_phonology_displays()

    def process_allophone_command_entry(self, args):
        # e.g. allophone m /mf/>/{ɱ}f/
        phoneme_symbol, rule_str = args
        assert phoneme_symbol in self.gui.language.phonemes

        inp, outp = rule_str.split(">")
        rule = Rule.from_input_and_output_strs(inp, outp, self.gui.language.symbol_dict)
        self.gui.language.phonemes[phoneme_symbol].add_allophone_rule(rule)

    def process_graph_command_entry(self, args):
        # e.g. graph a V
        grapheme_str, classes_str = args
        assert "<" not in grapheme_str and ">" not in grapheme_str, "graph command takes grapheme without <>"
        grapheme_str = "<" + grapheme_str + ">"
        grapheme = Grapheme.from_str(grapheme_str)
        classes = classes_str.split(",")
        classes = ["<"+c+">" for c in classes]
        self.orthography_converter.add_grapheme_to_classes(grapheme, classes)

    def process_pos_command_entry(self, args):
        # e.g. pos v {negation}-{tense}{person}{number}
        #      pos pa -
        #      pos pb {motion}-{number}
        pos, template = args
        # assert pos not in self.full_inflections_by_part_of_speech, "already have inflection template for pos \"{}\"".format(pos)
        self.gui.add_pos(pos)
        self.full_inflections_by_part_of_speech[pos] = []
        self.affixes_by_part_of_speech_and_feature[pos] = {}
        self.inflection_hierarchy_by_part_of_speech[pos] = []
        self.inflection_templates_by_part_of_speech[pos] = template
        if "{" in template:
            s1s = template.split("{")[1:]
            s2s = [s.split("}")[0] for s in s1s]
            # print("got features {}".format(s2s))
            self.affixes_by_part_of_speech_and_feature[pos] = {f: [] for f in s2s}
            # print(self.affixes_by_part_of_speech_and_feature)

    def process_inflect_command_entry(self, args):
        # e.g. inflect v ra = negation NEG
        #      inflect v ku = person 1
        #      inflect v \null = number SG
        pos, morpheme_string, equals_sign, feature, gloss = args
        if morpheme_string == "\\null":
            morpheme_string = ""
        # assert pos in self.full_inflections_by_part_of_speech and len(self.full_inflections_by_part_of_speech[pos]) > 0
        assert "\\" not in morpheme_string, "non-null morpheme cannot contain backslash, but this one does: {}".format(morpheme_string)
        assert equals_sign == "="
        morpheme = Morpheme(pos, feature, morpheme_string, gloss)
        try:
            self.affixes_by_part_of_speech_and_feature[pos][feature].append(morpheme)
            if feature not in self.inflection_hierarchy_by_part_of_speech[pos]:
                self.inflection_hierarchy_by_part_of_speech[pos].append(feature)
        except KeyError:
            print("can't access index [\"{}\"][\"{}\"] in the affix dict:\n{}"
                .format(pos, feature, self.affixes_by_part_of_speech_and_feature))
            return

        self.expand_templates_for_new_affix(morpheme)
        lexemes_of_pos = [lex for lex in self.gui.language.lexicon.lexemes if lex.part_of_speech == pos]
        inflection_forms = self.full_inflections_by_part_of_speech.get(pos, [])
        for lex in lexemes_of_pos:
            lex.create_forms(inflection_forms)
        self.gui.language.update_used_phonemes()
        self.gui.update_phonology_displays()
        self.gui.update_lexicon_displays()

    def process_lexeme_command_entry(self, args):
        # e.g. lex lahas = n mountain
        #      lex mak = v eat
        #      lex mo = pb g:out out, outside, out of
        le = args
        try:
            citation_form, *rest = le
            le_i = self.gui.language.lexicon.next_lexeme_designation
            citation_form = Word.from_str(citation_form, designation=str(le_i))
            assert all(x in self.gui.language.phonemes for x in citation_form.get_phonemes_used()), "undeclared phoneme in {}".format(citation_form)
            assert rest[0] == "=", "equals sign expected after lexeme, instead got: {}".format(rest[0])
            pos, *gloss = rest[1:]
            gloss = " ".join(gloss)
            assert pos in self.full_inflections_by_part_of_speech, "unknown part of speech \"{}\"; please declare it".format(pos)
            inflection_forms = self.full_inflections_by_part_of_speech.get(pos, [])
            lexeme = Lexeme(citation_form, pos, gloss, inflection_forms=inflection_forms)
            self.gui.language.lexicon.add_lexeme(lexeme)
            self.gui.language.update_used_phonemes()
            self.gui.update_phonology_displays()
            self.gui.update_lexicon_displays()
        except Exception as exc:
            print("This line does not appear to be valid: {}\nIt threw {}: {}".format(le, type(exc), exc))
            raise exc

    def process_sound_change_command_entry(self, sc):
        # e.g. sc ViV>VjV
        sc = " ".join(sc)  # in case there were more spaces in there that for some reason are supposed to be there, but processor split it on them
        rules = Rule.from_str(sc)
        for rule in rules:
            self.gui.apply_sound_change(rule)
        self.gui.language.update_used_phonemes()
        self.gui.update_phonology_displays()

    def process_ortho_command_entry(self, args):
        # e.g. ortho </#/llV> /#j<V>/
        grapheme_str, phoneme_str = args
        assert grapheme_str[0] == "<" and grapheme_str[-1] == ">"
        assert phoneme_str[0] == "/" and phoneme_str[-1] == "/"

        # replacement_rule = Rule.from_str("{}>{}".format(grapheme_str, phoneme_str), is_orthographic_rule=True, add_blanks=False)[0]  # rule is unidirectional, but should be able to use expansion this way and then extract both input and output from specific cases
        replacement_rule = Rule.from_input_and_output_strs(
            grapheme_str,
            phoneme_str,
            self.gui.language.symbol_dict,
        )
        print("reprule {}".format(replacement_rule))
        cases = replacement_rule.get_specific_cases(
            classes=self.gui.language.symbol_dict,
            used_phonemes=None,
        )
        if cases == []:
            raise RuntimeError("Got no cases of rule {}".format(replacement_rule))
        for r in cases:
            print("case: {}".format(r))
            g = r.get_input_str()
            p = r.get_output_str()
            self.orthography_converter.add_pair(g, p)

    def process_write_command_entry(self, args):
        # e.g. write /awi.a/
        phoneme_str, = args
        assert phoneme_str[0] == "/" and phoneme_str[-1] == "/"
        phoneme_str = phoneme_str[1:-1]
        res = self.orthography_converter.convert_phonemes_to_graphemes(phoneme_str)
        print(res)

    def process_read_command_entry(self, args):
        # e.g. read <aullha>
        grapheme_str, = args
        assert grapheme_str[0] == "<" and grapheme_str[-1] == ">"
        grapheme_str = grapheme_str[1:-1]
        res = self.orthography_converter.convert_graphemes_to_phonemes(grapheme_str)
        print(res)

    def process_time_command_entry(self, args):
        # e.g. time 2000 B.C.
        # e.g. time Edo Period
        # let the time designation be any string
        # but enforce that the order they are presented is chronological
        t = " ".join(args)
        assert t not in self.time_history
        self.time_history.append(t)

    def get_parts_of_speech(self):
        return sorted(self.full_inflections_by_part_of_speech.keys())

    def expand_templates_for_new_lexeme(self, lexeme):
        # is this even necessary? the lexeme will get its InflectionForms by looking up those in the full_inflections_... dict
        raise NotImplementedError

    def expand_templates_for_new_affix(self, morpheme):
        pos = morpheme.pos
        template = self.inflection_templates_by_part_of_speech[pos]
        # inflections = self.full_inflections_by_part_of_speech[pos]
        inflections = [InflectionForm(pos, template, gloss="", designation_suffix="")]
        for feature in self.inflection_hierarchy_by_part_of_speech[pos]:
            morphemes = self.affixes_by_part_of_speech_and_feature[pos][feature]
            assert type(morphemes) is list, morphemes
            new_inflections = []
            for inf in inflections:
                for morpheme_i, m in enumerate(morphemes):
                    assert type(m) is Morpheme, m
                    new_morpheme_string = inf.string.replace("{"+feature+"}", m.string)
                    new_gloss = inf.gloss + "-" + m.gloss
                    new_designation_suffix = inf.designation_suffix + "." + str(morpheme_i)
                    new_inflections.append(InflectionForm(pos, new_morpheme_string, new_gloss, new_designation_suffix))
            inflections = new_inflections
        self.full_inflections_by_part_of_speech[pos] = inflections

    def load_input_file(self, input_fp):
        extension = input_fp.split(".")[-1]
        if extension == "docx":
            return self.load_lexicon_from_docx(input_fp, command_processor)
        else:
            with open(input_fp) as f:
                lines = f.readlines()
            lines = [x.strip() for x in lines]
            for s in lines:
                self.process_command(s)
            return self.gui.language.lexicon

    def load_lexicon_from_docx(self, fp):
        document = docx.Document(fp)
        ps = [p.text for p in document.paragraphs]
        for s in ps:
            self.process_command(s)
        return self.gui.language.lexicon
