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
    assert n_words == n_words_outp, "mismatch in number of words"

    inputs = keras.Input(shape=(n_inp_features,))
    hidden = keras.layers.Dense(20, activation=tf.nn.relu)(inputs)
    outputs = keras.layers.Dense(n_outp_features, activation=tf.nn.softmax)(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse")
    model.fit(inp, outp, epochs=20)
    return model


def get_random_dataset(n_words, eng_semantic_model):
    # allow for frequency effects by drawing randomly from the corpus
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
    semantic_arr = w2v.convert_words_to_semantic_nn_input(eng_words, eng_semantic_model)
    # print(semantic_arr.shape)
    return eng_words, lang_words, articulation_arr, semantic_arr


if __name__ == "__main__":
    # allow for frequency effects by drawing randomly from the corpus
    eng_semantic_model = w2v.get_eng_model()
    eng_words = w2v.get_all_words_from_model(eng_semantic_model)

    eng_words_train, lang_words_train, phon_train, sem_train = get_random_dataset(1000, eng_semantic_model)
    eng_words_test, lang_words_test, phon_test, sem_test = get_random_dataset(10, eng_semantic_model)

    model_phon_to_sem = fit_nn(phon_train, sem_train)
    print(model_phon_to_sem)
    model_sem_to_phon = fit_nn(sem_train, phon_train)
    print(model_sem_to_phon)

    print("testing sound to meaning")
    predicted_meanings = model_phon_to_sem.predict(phon_test)
    for w, vec in zip(eng_words_test, predicted_meanings):
        print(f"\ntest word '{w}' has nearest neighbors:")
        d = w2v.get_nearest_neighbors([w], [vec], eng_semantic_model)
        for w2, dist in sorted(d[w].items(), key=lambda kv: kv[1]):
            print(f"'{w2}' at distance {dist}")

    print("testing meaning to sound")
    predicted_sounds = model_sem_to_phon.predict(sem_test)
