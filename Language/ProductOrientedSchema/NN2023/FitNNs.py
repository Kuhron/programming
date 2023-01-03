import random
import numpy as np
import matplotlib.pyplot as plt
import keras

import FitWord2Vec as w2v
import ArticulatoryFeatures as af
import RenderSentenceTemplates as rst


def fit_nn(inp, outp):
    n_words, n_inp_features = inp.shape
    n_words_outp, n_outp_features = outp.shape
    assert n_words == n_words_outp, "mismatch in number of words"
    input_layer = keras.Input(n_inp_features)
    model = keras.Model(input_layer, outp)
    



if __name__ == "__main__":
    # allow for frequency effects by drawing randomly from the corpus
    n_words = 1000

    translations = rst.get_translation_dict()
    eng_words = []
    eng_text_tokens = w2v.get_eng_text_tokenized()
    for i in range(n_words):
        sent = random.choice(eng_text_tokens)
        eng_word = random.choice(sent)
        eng_words.append(eng_word)
    lang_words = rst.translate_words(eng_words, translations)

    # want articulation in the language, but semantics from the unambiguous glosses
    articulation_arr = af.convert_words_to_articulatory_nn_input(lang_words)
    # print(articulation_arr.shape)
    semantic_arr = w2v.convert_words_to_semantic_nn_input(eng_words, eng_text_tokens)
    # print(semantic_arr.shape)

    nn_sound_to_meaning = fit_nn(articulation_arr, semantic_arr)
    nn_meaning_to_sound = fit_nn(semantic_arr, articulation_arr)
    