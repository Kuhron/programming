# treat words and cxs as the same thing
# make a lexicon of words and cxs
# each of these has membership in any of a bunch of word classes

import random
import numpy as np
import matplotlib.pyplot as plt

import JointDistributionMarkov as jdm


class Lexicon:
    def __init__(self):
        self.items = []
        self.items_by_class = {}
        self.concrete_items = []
        self.schematic_items = []

    def add_item(self, item):
        # add it to the items list
        self.items.append(item)

        # classify it as concrete (wordform) or schematic (cx)
        if item.is_schematic:
            self.schematic_items.append(item)
        else:
            self.concrete_items.append(item)

        # add it to the directory by word class, for filling out constructions more easily
        n_classes = len(item.word_class_vector)
        for i in range(n_classes):
            if i not in self.items_by_class:
                self.items_by_class[i] = []
            if item.word_class_vector[i]:
                self.items_by_class[i].append(item)

    @staticmethod
    def from_items(items):
        l = Lexicon()
        for item in items:
            l.add_item(item)
        return l

    @staticmethod
    def random(n, wordform_transitions, word_class_probabilities):
        items = [LexicalItem.random(wordform_transitions, word_class_probabilities) for i in range(n)]
        return Lexicon.from_items(items)


class LexicalItem:
    def __init__(self, form, word_class_vector):
        assert type(form) in [list, str], type(form)
        self.form = form
        self.is_schematic = type(form) is list
        self.word_class_vector = word_class_vector
        self.word_class_numbers = sorted([i for i, b in enumerate(self.word_class_vector) if b])

    @staticmethod
    def random(wordform_transitions, word_class_probabilities):
        wordform_probability = 0.8
        if random.random() < wordform_probability:
            # make an actual word
            form = get_wordform(wordform_transitions)
        else:
            # make a schematic cx
            n_classes = len(word_class_probabilities)
            cx = get_construction(n_classes)
            form = cx
        wclasses = get_word_class_vector(word_class_probabilities)
        return LexicalItem(form, wclasses)

    def __repr__(self):
        # return f"<Item {self.form} of classes {self.word_class_numbers}>"
        return f"{self.form}"


def get_word_class_probabilities():
    n_classes = random.randint(3, 15)
    p = np.random.uniform(0, 1, (n_classes,))
    # should skew it a bit so most classes are pretty rare and some are common (so some power > 1 that maps [0,1] to itself)
    power = 1 + abs(np.random.normal(0, 3))
    p = p ** power
    return p


def get_word_class_vector(probabilities):
    # array of 0 and 1 for if a word is in each of the word classes
    n_classes, = probabilities.shape
    rolls = np.random.uniform(0, 1, (n_classes,))
    membership = rolls < probabilities
    return membership


def get_wordform(transitions):
    return jdm.get_word(transitions)


def get_construction(n_classes):
    # make an array with some length, each thing is just a slot with a word class, see how that does
    get_num = lambda: random.randrange(n_classes)
    cx_len = random.randint(2, 4)
    cx = [get_num() for i in range(cx_len)]
    return cx


def convert_sentence_array_to_str(sent):
    sent = [x if type(x) is LexicalItem else convert_sentence_array_to_str(x) for x in sent]
    return " ".join(x if type(x) is str else x.form for x in sent)


def get_example_text(lexicon, wordform_transitions, word_class_probabilities):
    n_sentences = 100
    sentences = []
    for i in range(n_sentences):
        sent = get_example_sentence(lexicon, wordform_transitions, word_class_probabilities)
        sent_str = convert_sentence_array_to_str(sent)
        sentences.append(sent_str)
    s = "| " + " | ".join(sentences) + " |"
    return s


def get_example_sentence(lexicon, wordform_transitions, word_class_probabilities):
    # choose a schematic and then fill out everything
    # if you fail to get a form for a slot, generate a new item that can go in that slot (just force its bit for that slot's word class to be 1)
    cx = random.choice(lexicon.schematic_items)
    return fill_cx(cx, lexicon, wordform_transitions, word_class_probabilities)


def fill_cx(cx, lexicon, wordform_transitions, word_class_probabilities):
    assert cx.is_schematic
    slots = cx.form
    filled_slots = []
    for slot_word_class in slots:
        items_of_class = lexicon.items_by_class[slot_word_class]

        if len(items_of_class) == 0:
            # make a new one and force it to be in this class
            item = LexicalItem.random(wordform_transitions, word_class_probabilities)
            if not item.word_class_vector[slot_word_class]:
                item.word_class_vector[slot_word_class] = 1
                item.word_class_numbers = sorted([slot_word_class] + item.word_class_numbers)
                lexicon.add_item(item)
        else:
            item = random.choice(items_of_class)

        if item.is_schematic:
            filled_slot = fill_cx(item, lexicon, wordform_transitions, word_class_probabilities)
        else:
            filled_slot = item

        filled_slots.append(filled_slot)

    return filled_slots


if __name__ == "__main__":
    wordform_transitions = jdm.get_random_transition_possibilities()
    word_class_probabilities = get_word_class_probabilities()
    n_classes = len(word_class_probabilities)
    lexicon = Lexicon.random(100, wordform_transitions, word_class_probabilities)

    # text = get_example_text(lexicon, wordform_transitions, word_class_probabilities)
    # print(text)

    for i in range(100):
        print(get_example_sentence(lexicon, wordform_transitions, word_class_probabilities))
        print()
