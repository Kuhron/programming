from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QListWidget, QListWidgetItem, 
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

import sys


class Language:
    def __init__(self, lexicon):
        self.lexicon = lexicon


class Lexicon:
    def __init__(self, lexemes):
        self.lexemes = lexemes
    
    def add_lexeme(self, lexeme):
        self.lexemes.append(lexeme)


class Lexeme:
    def __init__(self, citation_form, forms):
        self.citation_form = citation_form
        self.forms = forms


class ConlangWorkspaceGUI(QDialog):
    def __init__(self, language, parent=None):
        super(ConlangWorkspaceGUI, self).__init__(parent)
        self.originalPalette = QApplication.palette()

        self.language = language
        mainLayout = QGridLayout()

        self.createLexemeList()
        mainLayout.addWidget(self.lexeme_list)

        self.createLexemeFormList()
        mainLayout.addWidget(self.lexeme_form_list)
        self.clearSelectedLexeme()

        self.setLayout(mainLayout)

        self.setWindowTitle("Conlang Workspace")

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