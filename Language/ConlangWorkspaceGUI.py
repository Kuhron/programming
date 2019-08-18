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

from LanguageEvolutionTools import (Word, Rule, Lexeme, Lexicon, Language,
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

        lexiconTab = QWidget()
        self.createLexemeList()
        self.createLexemeFormList()
        # self.clearSelectedLexeme()
        export_to_docx_button = QPushButton("Export to DOCX")
        export_to_docx_button.pressed.connect(self.export_lexicon_to_docx)

        lexiconTabHBox = QHBoxLayout()
        lexiconTabHBox.setContentsMargins(5, 5, 5, 5)
        lexiconTabHBox.addWidget(self.lexeme_list)
        lexiconTabHBox.addWidget(self.lexeme_form_list)
        lexiconTabHBox.addWidget(export_to_docx_button)
        lexiconTab.setLayout(lexiconTabHBox)

        soundChangeTab = QWidget()
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
        soundChangeTab.setLayout(soundChangeTabHBox)

        terminalTab = QWidget()
        terminalInputWidget = QLineEdit()
        terminalOutputWidget = QTextEdit()
        terminalTabHBox = QHBoxLayout()
        terminalTabHBox.setContentsMargins(5, 5, 5, 5)
        terminalTabHBox.addWidget(terminalInputWidget)
        terminalTabHBox.addWidget(terminalOutputWidget)
        terminalTab.setLayout(terminalTabHBox)

        self.tabWidget.addTab(lexiconTab, "Lexicon")
        self.tabWidget.addTab(soundChangeTab, "Sound Changes")
        self.tabWidget.addTab(terminalTab, "Terminal")

    def createLexemeList(self):
        self.lexeme_list = QListWidget()
        self.populateLexemeList()
        self.lexeme_list.currentItemChanged.connect(self.changeSelectedLexeme)

    def populateLexemeList(self):
        for lex in self.language.lexicon.lexemes:
            item = QListWidgetItem(lex.citation_form.to_str() + " (" + lex.gloss + ")")
            item.setData(Qt.UserRole, lex)
            self.lexeme_list.addItem(item)

    def createLexemeFormList(self):
        self.lexeme_form_list = QListWidget()

    def clearSelectedLexeme(self):
        # doesn't seem to work, qt likes to trigger the currentItemChanged signal anyway
        self.lexeme_list.clearSelection()
        self.lexeme_form_list.clear()

    def changeSelectedLexeme(self):
        self.lexeme_form_list.clear()
        lex = self.lexeme_list.currentItem().data(Qt.UserRole)
        for form, form_gloss in lex.form_to_gloss.items():
            self.lexeme_form_list.addItem(form + " (" + form_gloss + ")")

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
        rules = Rule.from_str(self.soundChangeWidget.text())
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
            new_lexeme = Lexeme(new_citation_form, new_forms, lexeme.part_of_speech, lexeme.gloss, lexeme.form_glosses)
            new_lexicon.add_lexeme(new_lexeme)
        self.lexicon = new_lexicon
        self.update_lexicon_displays()

    def update_lexicon_displays(self):
        self.populateLexemeList()


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
    for ce in command_entries:
        command, *rest = ce.split(" ")
        assert command[0] == "\\"
        command = command[1:]
        # print("got command {} with args {}".format(command, rest))
        if command == "pos":
            pos, template = rest
            assert pos not in full_inflections_by_part_of_speech, "already have inflection template for pos \"{}\"".format(pos)
            full_inflections_by_part_of_speech[pos] = {template: ""}  # inflection: gloss, and gloss will be appended to in expansion phase
            affixes_by_part_of_speech_and_feature[pos] = {}
            if "{" in template:
                s1s = template.split("{")[1:]
                s2s = [s.split("}")[0] for s in s1s]
                # print("got features {}".format(s2s))
                affixes_by_part_of_speech_and_feature[pos] = {f: {} for f in s2s}
                # print(affixes_by_part_of_speech_and_feature)

        elif command == "inflect":
            pos, morpheme, equals_sign, feature, gloss = rest
            if morpheme == "\\null":
                morpheme = ""
            # assert pos in full_inflections_by_part_of_speech and len(full_inflections_by_part_of_speech[pos]) > 0
            assert "\\" not in morpheme, "morpheme cannot contain backslash, but this one does: {}".format(morpheme)
            assert equals_sign == "="
            try:
                affixes_by_part_of_speech_and_feature[pos][feature][morpheme] = gloss
            except KeyError:
                print("can't access index [\"{}\"][\"{}\"] in the affix dict:\n{}"
                    .format(pos, feature, affixes_by_part_of_speech_and_feature))
    
    # print(affixes_by_part_of_speech_and_feature)
    # print(full_inflections_by_part_of_speech)
    for pos in affixes_by_part_of_speech_and_feature:
        # expand templates
        inflections = full_inflections_by_part_of_speech[pos]
        for feature in affixes_by_part_of_speech_and_feature[pos]:
            morphemes = affixes_by_part_of_speech_and_feature[pos][feature]
            assert type(morphemes) is dict, morphemes
            new_inflections = {}
            for inf, inf_gloss in inflections.items():
                for m, m_gloss in morphemes.items():
                    new_inf = inf.replace("{"+feature+"}", m)
                    new_gloss = inf_gloss + "-" + m_gloss
                    new_inflections[new_inf] = new_gloss
            inflections = new_inflections
        full_inflections_by_part_of_speech[pos] = inflections
    # print(full_inflections_by_part_of_speech)
    
    for le_i, le in enumerate(lexeme_entries):
        try:
            citation_form, le = le.split(" = ")
            citation_form = Word.from_str(citation_form, str(le_i))
            # print("citation form:", citation_form)
            pos, *gloss = le.split(" ")
            gloss = " ".join(gloss)
            assert pos in full_inflections_by_part_of_speech, "unknown part of speech \"{}\"; please declare it"
            affix_to_gloss = full_inflections_by_part_of_speech.get(pos, {})
            forms = []
            form_glosses = []
            for affix, inflection_gloss in affix_to_gloss.items():
                forms.append(inflect_lexeme(citation_form, affix))
                form_glosses.append(inflect_gloss(gloss, inflection_gloss))
                # for inflection_gloss in full_inflections_by_part_of_speech[pos].values()]
            lexeme = Lexeme(citation_form, forms, pos, gloss, form_glosses)
            lexicon.add_lexeme(lexeme)
        except Exception as exc:
            print("This line does not appear to be valid: {}\nIt threw {}: {}".format(le, type(exc), exc))
            raise exc
    return lexicon


def inflect_lexeme(citation_form, inflection):
    # later, implement more robust logic
    # affix should be able to be a string representing a function of citation_form
    # e.g. citation form is k_t_b and affix is _i_a_
    return inflection.replace("_", citation_form.to_str())

def inflect_gloss(citation_gloss, inflection_gloss):
    return citation_gloss + "-" + inflection_gloss


if __name__ == '__main__':
    lexicon = load_lexicon_from_docx("/home/wesley/programming/Language/ExamplishLexiconDocx.docx")
    # lexicon = Lexicon([])
    # roots = [
    #     "koh", "tom", "asur", "puram", "mit", "ul", "nir", 
    #     "kal", "lahas", "alat", "eril", "erek", "iliandir", "pahal",
    #     "ettel", "risam", "tamas", "issol", "maruk", "koratin", "sertem",
    # ]
    # num_suffixes = ["", "ail"]
    # case_suffixes = ["", "en", "ak"]
    # glosses = [
    #     "dog", "person", "fish", "house", "hand", "fire", "water",
    #     "mountain", "storm", "ground", "river", "lake", "animal", "wood",
    #     "salt", "metal", "sand", "leaf", "knife", "nut", "tool",
    # ]
    # suffixes = [n+c for n in num_suffixes for c in case_suffixes]
    # for root, gloss in zip(roots, glosses):
    #     forms = [root+s for s in suffixes]
    #     part_of_speech = "n"
    #     lex = Lexeme(root, forms, part_of_speech, gloss)
    #     lexicon.add_lexeme(lex)

    language = Language("Examplish", lexicon, DEFAULT_PHONEME_CLASSES)

    app = QApplication(sys.argv)
    gui = ConlangWorkspaceGUI(language)
    gui.show()
    sys.exit(app.exec_())