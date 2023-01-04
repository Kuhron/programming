import random
import numpy as np
import matplotlib.pyplot as plt

# suppress annoying tensorflow messages about GPUs and shared objects and crap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras

import FitWord2Vec as w2v
import ArticulatoryFeatures as af
import RenderSentenceTemplates as rst


def fit_nn(inp, outp):
    n_words, n_inp_features = inp.shape
    n_words_outp, n_outp_features = outp.shape
    assert n_words == n_words_outp, f"mismatch in number of words: input has {n_words}, output has {n_words_outp}"

    # TODO think about the actual model structure, loss, hyperparams, etc.
    # do they actually make sense for what you're trying to do?
    # in one direction: given a semantic embedding vector, predict articulation array
    # in the other: given an articulation array, predict embedding vector
    # need to make sure hidden layer doesn't destroy information by being too small
    # what activation functions make sense?
    inputs = keras.Input(shape=(n_inp_features,))
    hidden = keras.layers.Dense(500, activation=tf.nn.relu)(inputs)
    outputs = keras.layers.Dense(n_outp_features, activation=None)(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="mean_absolute_error")
    model.fit(inp, outp, epochs=25)
    return model


def get_random_dataset(n_words, eng_word_to_vector_dict):
    # allow for frequency effects by drawing randomly from the corpus
    translations = rst.get_translation_dict()
    eng_words = []
    eng_text_tokens = w2v.get_eng_text_tokenized()
    for i in range(n_words):
        sent = random.choice(eng_text_tokens)
        eng_word = random.choice(sent)
        eng_words.append(eng_word)
    lang_words = rst.translate_word_glosses(eng_words, translations)

    # want articulation in the language, but semantics from the unambiguous glosses
    articulation_arr = af.convert_words_to_articulatory_nn_input(lang_words)
    # print(articulation_arr.shape)
    semantic_arr = w2v.convert_words_to_semantic_nn_input(eng_words, eng_word_to_vector_dict)
    # print(semantic_arr.shape)
    return eng_words, lang_words, articulation_arr, semantic_arr



if __name__ == "__main__":
    # allow for frequency effects by drawing randomly from the corpus
    translation_dict = rst.get_translation_dict()
    eng_text_tokens = w2v.get_eng_text_tokenized()
    wv = w2v.get_word_to_vector_dict_combined_model(eng_text_tokens, vector_size=100, min_window=3, max_window_incl=13, sg=True)

    eng_words_train, lang_words_train, phon_train, sem_train = get_random_dataset(100000, wv)
    eng_words_test, lang_words_test, phon_test, sem_test = get_random_dataset(25, wv)

    print("fitting model of sound to meaning")
    model_phon_to_sem = fit_nn(phon_train, sem_train)
    print("fitting model of meaning to sound")
    model_sem_to_phon = fit_nn(sem_train, phon_train)

    print("testing sound to meaning")
    predicted_meanings = model_phon_to_sem.predict(phon_test)
    for w, vec in zip(eng_words_test, predicted_meanings):
        print(f"\ntest word '{w}' has predicted meaning vector with nearest neighbors:")
        d = w2v.get_nearest_neighbors([w], [vec], wv)
        for w2, dist in sorted(d[w].items(), key=lambda kv: kv[1]):
            print(f"'{w2}' at distance {dist}")
        print("its actual nearest neighbors in the embedding space are:")
        real_vec = wv[w]
        d = w2v.get_nearest_neighbors([w], [real_vec], wv)
        for w2, dist in sorted(d[w].items(), key=lambda kv: kv[1]):
            print(f"'{w2}' at distance {dist}")

    print("testing meaning to sound")
    predicted_sounds = model_sem_to_phon.predict(sem_test)
    for w, vec in zip(eng_words_test, predicted_sounds):
        lang_w = rst.translate_word_gloss(w, translation_dict)
        print(f"\ntest word {w} translates to {lang_w}")
        arr = vec.reshape(af.ARRAY_SHAPE)
        af.plot_articulatory_array(arr, label=f"predictions/{w}_{lang_w}")
    # make a plot of how the articulations are made in the predicted wordform, see what existing phones it sounds like
