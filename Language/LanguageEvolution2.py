import codecs
import itertools
import os
import pickle
import random

from copy import deepcopy

from LanguageEvolutionTools import *


if __name__ == "__main__":
    lexicon = LE2Lexicon.from_user_input()
    inventory = Inventory.from_lexicon(lexicon)
    
    # check that the original routine still works after refactoring stuff
    generate_language_and_write_to_file()
