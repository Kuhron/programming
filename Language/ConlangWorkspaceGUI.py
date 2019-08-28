from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, 
        QInputDialog, QLabel, QLineEdit,
        QListWidget, QListWidgetItem, 
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget
    )

import docx
from docx.shared import Pt

from LanguageEvolutionTools import (
        Word, Rule, Lexeme, Lexicon, Language, InflectionForm, Morpheme, 
        DEFAULT_PHONEME_CLASSES, 
        evolve_word, get_random_rules, 
    )

import sys


class ConlangWorkspaceGUI(QDialog):
    def __init__(self, language, parent=None):
        super(ConlangWorkspaceGUI, self).__init__(parent)
        self.originalPalette = QApplication.palette()

        self.language = language
        mainLayout = QGridLayout()

        self.createTabWidget()
        mainLayout.addWidget(self.tabWidget)

        self.setLayout(mainLayout)

        self.setWindowTitle("Conlang Workspace")

    def createTabWidget(self):
        self.tabWidget = QTabWidget()
        # self.tabWidget.setSizePolicy(QSizePolicy.Preferred,
        #         QSizePolicy.Ignored)

        self.lexiconTab = QWidget()
        self.createLexemeList()
        self.createLexemeFormList()
        # self.clearSelectedLexeme()
        create_sound_change_button = QPushButton("Create sound change")
        create_sound_change_button.pressed.connect(self.create_sound_change_from_word)
        export_to_docx_button = QPushButton("Export to DOCX")
        export_to_docx_button.pressed.connect(self.export_lexicon_to_docx)

        lexiconTabHBox = QHBoxLayout()
        lexiconTabHBox.setContentsMargins(5, 5, 5, 5)
        lexiconTabHBox.addWidget(self.lexeme_list)
        lexiconTabHBox.addWidget(self.lexeme_form_list)
        lexiconTabHBox.addWidget(create_sound_change_button)
        lexiconTabHBox.addWidget(export_to_docx_button)
        self.lexiconTab.setLayout(lexiconTabHBox)

        self.soundChangeTab = QWidget()
        self.soundChangeWidget = QLineEdit()
        soundChangeLabel = QLabel("Rule")
        soundChangeLabel.setBuddy(self.soundChangeWidget)
        soundChangeGenerateButton = QPushButton("Generate rule")
        soundChangeGenerateButton.pressed.connect(self.generate_sound_change)
        applySoundChangeButton = QPushButton("Apply rule")
        applySoundChangeButton.pressed.connect(self.apply_sound_change)

        soundChangeTabHBox = QHBoxLayout()
        soundChangeTabHBox.setContentsMargins(5, 5, 5, 5)
        soundChangeTabHBox.addWidget(self.soundChangeWidget)
        soundChangeTabHBox.addWidget(soundChangeGenerateButton)
        soundChangeTabHBox.addWidget(applySoundChangeButton)
        self.soundChangeTab.setLayout(soundChangeTabHBox)

        self.terminalTab = QWidget()
        terminalInputWidget = QLineEdit()
        terminalOutputWidget = QTextEdit()
        terminalTabHBox = QHBoxLayout()
        terminalTabHBox.setContentsMargins(5, 5, 5, 5)
        terminalTabHBox.addWidget(terminalInputWidget)
        terminalTabHBox.addWidget(terminalOutputWidget)
        self.terminalTab.setLayout(terminalTabHBox)

        self.tabWidget.addTab(self.lexiconTab, "Lexicon")
        self.tabWidget.addTab(self.soundChangeTab, "Sound Changes")
        self.tabWidget.addTab(self.terminalTab, "Terminal")

    def createLexemeList(self):
        self.lexeme_list = QListWidget()
        self.populateLexemeList()
        self.lexeme_list.currentItemChanged.connect(self.changeSelectedLexeme)

    def populateLexemeList(self):
        for lex in self.language.lexicon.lexemes:
            label = lex.citation_form.to_str() + " ({}, {})".format(lex.part_of_speech, lex.gloss)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, lex)
            self.lexeme_list.addItem(item)

    def createLexemeFormList(self):
        self.lexeme_form_list = QListWidget()
        # self.lexeme_form_list.currentItemChanged.connect(self.report_current_selected_form)

    def report_current_selected_form(self):
        item = self.lexeme_form_list.currentItem()
        if item is None:
            return
        w = item.data(Qt.UserRole)
        print(w)

    def clearSelectedLexeme(self):
        self.lexeme_list.clearSelection()
        self.lexeme_form_list.clear()

    def changeSelectedLexeme(self):
        self.lexeme_form_list.clear()
        item = self.lexeme_list.currentItem()
        if item is None:
            return
        lex = item.data(Qt.UserRole)
        sorted_forms = sorted(lex.forms, key=lambda w: w.designation)
        # for form, form_gloss in lex.form_to_gloss.items():
        for form in sorted_forms:
            assert type(form) is Word
            item = QListWidgetItem(form.to_str() + " (" + form.gloss + ")")
            item.setData(Qt.UserRole, form)
            self.lexeme_form_list.addItem(item)

    def export_lexicon_to_docx(self):
        # TODO include all command lines
        # output_fp = "/home/wesley/programming/DocxTestOutput.docx"
        output_fp = "/home/wesley/programming/Language/ExamplishLexiconDocx_GENERATED.docx"
        lexicon = self.language.lexicon

        document = docx.Document()

        style = document.styles["Normal"]
        font = style.font
        font.name = "Charis SIL"
        font.size = Pt(12)

        document.add_paragraph(self.language.name + " Lexicon")
        document.add_paragraph()

        for lexeme in lexicon.lexemes:
            p = document.add_paragraph()
            p.add_run(lexeme.citation_form)
            p.add_run(" = ")
            p.add_run(lexeme.part_of_speech).italic = True
            p.add_run(" " + lexeme.gloss)

        # handle irregularities, notes, etc. after this

        document.save(output_fp)

    def generate_sound_change(self):
        rule = get_random_rules(1, self.language.lexicon.all_forms(), self.language.phoneme_classes)[0]
        self.soundChangeWidget.setText(rule.to_notation())

    def apply_sound_change(self):
        rules = Rule.from_str(self.soundChangeWidget.text())#.replace("Ø", ""))  # shouldn't have null symbols anymore
        expanded_rules = []
        for rule in rules:
            expanded_rules += rule.get_specific_cases(self.language.phoneme_classes, self.language.used_phonemes)
        new_lexicon = Lexicon([])
        for lexeme in self.language.lexicon.lexemes:
            new_citation_form = evolve_word(lexeme.citation_form, expanded_rules)
            new_forms = []
            for f in lexeme.forms:
                new_form = evolve_word(f, expanded_rules)
                new_forms.append(new_form)
            new_lexeme = Lexeme(new_citation_form, lexeme.part_of_speech, lexeme.gloss, forms=new_forms)
            new_lexicon.add_lexeme(new_lexeme)
        self.language.lexicon = new_lexicon
        self.update_lexicon_displays()
        self.soundChangeWidget.clear()
        print("sound change(s) applied: {}".format(rules))

    def update_lexicon_displays(self):
        self.clearSelectedLexeme()
        self.lexeme_list.clear()
        self.lexeme_form_list.clear()
        self.populateLexemeList()

    def create_sound_change_from_word(self):
        item = self.lexeme_form_list.currentItem()
        if item is None:
            item = self.lexeme_list.currentItem()
            if item is None:
                print("no item selected to create sound change from")
                return
        w = item.data(Qt.UserRole)
        assert type(w) is Word
        self.tabWidget.setCurrentWidget(self.soundChangeTab)
        self.soundChangeWidget.setText(w.to_str())


class CommandProcessor:
    def __init__(self):
        self.command_history = []
    
    def process_command(command_str):
        command_str = command_str.strip()
        if command_str == "" or command_str[0] == "#":
            # do nothing on blank lines and comments
            return

        # if command was processed successfully
        self.command_history.append(command_str)


def load_lexicon_from_docx(fp):
    lexicon = Lexicon([])
    document = docx.Document(fp)
    ps = [p.text for p in document.paragraphs]
    ps = [p for p in ps if len(p) > 0]
    is_lexeme_entry = lambda p: p[0] not in ["\\", "#"] and " = " in p
    # use backslash to start special lines, such as defining a word class
    # use '#' for comment lines
    is_command_entry = lambda p: p[0] == "\\"
    lexeme_entries = [p for p in ps if is_lexeme_entry(p)]
    command_entries = [p for p in ps if is_command_entry(p)]

    full_inflections_by_part_of_speech = {}
    affixes_by_part_of_speech_and_feature = {}
    inflection_hierarchy_by_part_of_speech = {}
    for ce in command_entries:
        command, *rest = ce.split(" ")
        assert command[0] == "\\"
        command = command[1:]
        # print("got command {} with args {}".format(command, rest))
        if command == "pos":
            pos, template = rest
            assert pos not in full_inflections_by_part_of_speech, "already have inflection template for pos \"{}\"".format(pos)
            full_inflections_by_part_of_speech[pos] = [InflectionForm(pos, template, gloss="", designation_suffix="")]
            affixes_by_part_of_speech_and_feature[pos] = {}
            inflection_hierarchy_by_part_of_speech[pos] = []
            if "{" in template:
                s1s = template.split("{")[1:]
                s2s = [s.split("}")[0] for s in s1s]
                # print("got features {}".format(s2s))
                affixes_by_part_of_speech_and_feature[pos] = {f: [] for f in s2s}
                # print(affixes_by_part_of_speech_and_feature)

        elif command == "inflect":
            pos, morpheme_string, equals_sign, feature, gloss = rest
            if morpheme_string == "\\null":
                morpheme_string = ""
            # assert pos in full_inflections_by_part_of_speech and len(full_inflections_by_part_of_speech[pos]) > 0
            assert "\\" not in morpheme_string, "morpheme cannot contain backslash, but this one does: {}".format(morpheme_string)
            assert equals_sign == "="
            morpheme = Morpheme(pos, feature, morpheme_string, gloss)
            try:
                affixes_by_part_of_speech_and_feature[pos][feature].append(morpheme)
                if feature not in inflection_hierarchy_by_part_of_speech[pos]:
                    inflection_hierarchy_by_part_of_speech[pos].append(feature)
            except KeyError:
                print("can't access index [\"{}\"][\"{}\"] in the affix dict:\n{}"
                    .format(pos, feature, affixes_by_part_of_speech_and_feature))
    
    for pos in affixes_by_part_of_speech_and_feature:
        # expand templates
        inflections = full_inflections_by_part_of_speech[pos]
        for feature in inflection_hierarchy_by_part_of_speech[pos]:
            morphemes = affixes_by_part_of_speech_and_feature[pos][feature]
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
        full_inflections_by_part_of_speech[pos] = inflections
    
    for le_i, le in enumerate(lexeme_entries):
        try:
            citation_form, rest = le.split(" = ")
            citation_form = Word.from_str(citation_form, designation=str(le_i))
            pos, *gloss = rest.split(" ")
            gloss = " ".join(gloss)
            assert pos in full_inflections_by_part_of_speech, "unknown part of speech \"{}\"; please declare it"
            inflection_forms = full_inflections_by_part_of_speech.get(pos, [])
            lexeme = Lexeme(citation_form, pos, gloss, inflection_forms=inflection_forms)
            lexicon.add_lexeme(lexeme)
        except Exception as exc:
            print("This line does not appear to be valid: {}\nIt threw {}: {}".format(le, type(exc), exc))
            raise exc
    return lexicon


if __name__ == '__main__':
    lexicon = load_lexicon_from_docx("/home/wesley/programming/Language/ExamplishLexiconDocx.docx")
    language = Language("Examplish", lexicon, DEFAULT_PHONEME_CLASSES)

    app = QApplication(sys.argv)
    gui = ConlangWorkspaceGUI(language)
    gui.show()
    sys.exit(app.exec_())