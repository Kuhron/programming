from InflectionForm import InflectionForm
from Lexeme import Lexeme
from Morpheme import Morpheme
from Rule import Rule
from Word import Word
from OrthographyConverter import OrthographyConverter

import docx
from docx.shared import Pt


class CommandProcessor:
    def __init__(self, gui, orthography_converter):
        self.command_history = []
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
        return s[0] == "\\"

    @staticmethod
    def is_lexeme_entry(s):
        return s[0] not in ["\\", "#"] and " = " in s
        # use backslash to start special lines, such as defining a word class
        # use '#' for comment lines
    
    def process_command(self, command_str):
        command_str = command_str.strip()
        # if command_str == "" or command_str[0] == "#":
        #     # do nothing on blank lines and comments
        #     return
        # print("command to process:", command_str)

        if command_str == "" or command_str[0] == "#":
            pass
        elif CommandProcessor.is_command_entry(command_str):
            self.process_command_entry(command_str)
        elif CommandProcessor.is_lexeme_entry(command_str):
            self.process_lexeme_entry(command_str)
            # self.expand_templates_for_new_lexeme()
        else:
            print("command cannot be processed: {}".format(command_str))
            return

        # update the lexicon based on any new features, new affixes for a feature, new lexemes, etc.
        # self.expand_templates()  # crashes computer!

        # if command was processed successfully
        self.command_history.append(command_str)
        self.gui.update_command_history_display()

    def process_command_entry(self, ce):
        command, *rest = ce.split(" ")
        assert command[0] == "\\"
        command = command[1:]
        print("got command {} with args {}".format(command, rest))

        if False:
            pass  # just so I don't have to keep changing "if" to "elif" when adding new commands
        elif command == "include":
            assert len(rest) == 1
            self.load_input_file(rest[0])
        elif command == "phone":
            self.process_phone_command_entry(rest)
        elif command == "pos":
            self.process_pos_command_entry(rest)
            # don't expand templates here because only declaring new pos, no inflections yet
        elif command == "inflect":
            self.process_inflect_command_entry(rest)
            # templates for this pos will be expanded in the process_inflect_... method
        elif command == "sc":
            self.process_sound_change_command_entry(rest)
        elif command == "graph":
            self.process_graph_command_entry(rest)
        elif command == "ortho":
            self.process_ortho_command_entry(rest)
        elif command == "write":
            self.process_write_command_entry(rest)
        elif command == "read":
            self.process_read_command_entry(rest)
        else:
            print("unknown command \'{}\'".format(command))

    def process_phone_command_entry(self, args):
        phoneme_symbol, classes_str = args
        classes = classes_str.split(",")
        self.gui.language.add_phoneme(phoneme_symbol, classes)
        self.gui.update_phonology_displays()

    def process_graph_command_entry(self, args):
        grapheme, classes_str = args
        classes = classes_str.split(",")
        self.orthography_converter.add_grapheme_to_classes(grapheme, classes)

    def process_pos_command_entry(self, args):
        # e.g. \pos v {negation}_{tense}{person}{number}
        #      \pos pa _
        #      \pos pb {motion}_{number}
        pos, template = args
        assert pos not in self.full_inflections_by_part_of_speech, "already have inflection template for pos \"{}\"".format(pos)
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
        # e.g. \inflect v ra = negation NEG
        #      \inflect v ku = person 1
        #      \inflect v \null = number SG
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

    def process_lexeme_entry(self, le):
        # e.g. lahas = n mountain
        #      mak = v eat
        #      mo = pb g:out out, outside, out of
        try:
            citation_form, rest = le.split(" = ")
            le_i = self.gui.language.lexicon.next_lexeme_designation
            citation_form = Word.from_str(citation_form, designation=str(le_i))
            pos, *gloss = rest.split(" ")
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
        sc = " ".join(sc)  # in case there were more spaces in there that for some reason are supposed to be there, but processor split it on them
        rules = Rule.from_str(sc)
        for rule in rules:
            self.gui.apply_sound_change(rule)
        self.gui.language.update_used_phonemes()
        self.gui.update_phonology_displays()

    def process_ortho_command_entry(self, args):
        grapheme_str, phoneme_str = args
        assert grapheme_str[0] == "<" and grapheme_str[-1] == ">"
        grapheme_str = grapheme_str[1:-1]
        assert phoneme_str[0] == "/" and phoneme_str[-1] == "/"
        phoneme_str = phoneme_str[1:-1]

        # replacement_rule = Rule.from_str("{}>{}".format(grapheme_str, phoneme_str), is_orthographic_rule=True, add_blanks=False)[0]  # rule is unidirectional, but should be able to use expansion this way and then extract both input and output from specific cases
        replacement_rule = Rule.from_input_and_output_strs(
            grapheme_str,
            phoneme_str,
            is_orthographic_rule=True,
            grapheme_classes=self.orthography_converter.grapheme_classes
        )
        cases = replacement_rule.get_specific_cases(
            phoneme_classes=self.gui.language.phoneme_classes,
            grapheme_classes=self.orthography_converter.grapheme_classes,
            used_phonemes=None
        )
        if cases == []:
            raise RuntimeError("Got no cases of rule {}".format(replacement_rule))
        for r in cases:
            print("case: {}".format(r))
            g = r.get_input_str()
            p = r.get_output_str()
            self.orthography_converter.add_pair(g, p)

    def process_write_command_entry(self, args):
        phoneme_str, = args
        assert phoneme_str[0] == "/" and phoneme_str[-1] == "/"
        phoneme_str = phoneme_str[1:-1]
        res = self.orthography_converter.convert_phonemes_to_graphemes(phoneme_str)
        print(res)

    def process_read_command_entry(self, args):
        grapheme_str, = args
        assert grapheme_str[0] == "<" and grapheme_str[-1] == ">"
        grapheme_str = grapheme_str[1:-1]
        res = self.orthography_converter.convert_graphemes_to_phonemes(grapheme_str)
        print(res)

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
