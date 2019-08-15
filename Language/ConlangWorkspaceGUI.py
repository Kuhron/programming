from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QListWidget, QListWidgetItem, 
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

from LanguageEvolutionTools import Word, Rule, Lexeme, Lexicon, Language

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

        lexiconTabHBox = QHBoxLayout()
        lexiconTabHBox.setContentsMargins(5, 5, 5, 5)
        lexiconTabHBox.addWidget(self.lexeme_list)
        lexiconTabHBox.addWidget(self.lexeme_form_list)
        lexiconTab.setLayout(lexiconTabHBox)

        soundChangeTab = QWidget()

        textEdit = QTextEdit()
        textEdit.setPlainText("asdf")

        soundChangeTabHBox = QHBoxLayout()
        soundChangeTabHBox.setContentsMargins(5, 5, 5, 5)
        soundChangeTabHBox.addWidget(textEdit)
        soundChangeTab.setLayout(soundChangeTabHBox)

        self.tabWidget.addTab(lexiconTab, "Lexicon")
        self.tabWidget.addTab(soundChangeTab, "Sound Changes")

    def createLexemeList(self):
        self.lexeme_list = QListWidget()
        for lex in self.language.lexicon.lexemes:
            item = QListWidgetItem(lex.citation_form)
            item.setData(Qt.UserRole, lex)
            self.lexeme_list.addItem(item)
        self.lexeme_list.currentItemChanged.connect(self.changeSelectedLexeme)

    def createLexemeFormList(self):
        self.lexeme_form_list = QListWidget()

    def clearSelectedLexeme(self):
        # doesn't seem to work, qt likes to trigger the currentItemChanged signal anyway
        self.lexeme_list.clearSelection()
        self.lexeme_form_list.clear()

    def changeSelectedLexeme(self):
        self.lexeme_form_list.clear()
        lex = self.lexeme_list.currentItem().data(Qt.UserRole)
        for form in lex.forms:
            self.lexeme_form_list.addItem(form)


if __name__ == '__main__':
    lexicon = Lexicon([])
    roots = [
        "koh", "tom", "asur", "puram", "mit", "ul", "nir", 
        "kal", "lahas", "alat", "eril", "erek", "iliandir", "pahal",
        "ettel", "risam", "tamas", "issol", "maruk", "koratin", "sertem",
    ]
    num_suffixes = ["", "ail"]
    case_suffixes = ["", "en", "ak"]
    suffixes = [n+c for n in num_suffixes for c in case_suffixes]
    for root in roots:
        forms = [root+s for s in suffixes]
        lex = Lexeme(root, forms)
        lexicon.add_lexeme(lex)

    language = Language(lexicon)

    app = QApplication(sys.argv)
    gui = ConlangWorkspaceGUI(language)
    gui.show()
    sys.exit(app.exec_())