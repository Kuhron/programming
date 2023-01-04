# adapted from https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

import gensim
import random
import re
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors

import RenderSentenceTemplates as rst

eng_sample_fp = "eng_sample.txt"
lang_sample_fp = "lang_sample.txt"


def sent_tokenize(s):
    pattern = "[^.]+\."
    return re.findall(pattern, s)


def word_tokenize(s):
    pattern = "[^\s]+"
    return re.findall(pattern, s)


def get_all_words_from_text_tokens(text_tokens):
    all_words = set()
    for sent in text_tokens:
        all_words |= set(word for word in sent)
    return sorted(all_words)


def print_all_word_vectors(all_words, cbow_model, sg_model):
    for w in all_words:
        cbow_vec = cbow_model.wv[w]
        sg_vec = sg_model.wv[w]
        print(f"\n word {w} has\nCBOW vector {cbow_vec}\nSG vector {sg_vec}")


def get_eng_text_tokenized():
    fp = eng_sample_fp
    return get_text_tokenized(fp)


def get_lang_text_tokenized():
    fp = lang_sample_fp
    return get_text_tokenized(fp)


def get_text_tokenized(fp):
    with open(fp) as f:
        lines = f.readlines()

    # put each sentence on its own line in the file so "\n" acts as the sentence delimiter
    sentences = [l.strip() for l in lines]
    sentences = [x for x in sentences if len(x) > 0]
    text_tokens = []
    for sent in sentences:
        sent_tokens = []
        for word in word_tokenize(sent):
            sent_tokens.append(word.lower())
        text_tokens.append(sent_tokens)
    return text_tokens


def convert_words_to_semantic_nn_input(words, word_to_vector_dict):
    n_words = len(words)
    n_cols = len(word_to_vector_dict[words[0]])
    arr = np.zeros((n_words, n_cols))
    for i, w in enumerate(words):
        vec = word_to_vector_dict[w]
        assert len(vec) == n_cols
        arr[i,:] = vec
    return arr


def get_eng_model(vector_size=100, window=5, sg=True):
    text_tokens = get_eng_text_tokenized()
    return get_model(text_tokens, vector_size, window, sg)


def get_lang_model(vector_size=100, window=5, sg=True):
    text_tokens = get_lang_text_tokenized()
    return get_model(text_tokens, vector_size, window, sg)


def get_model(text_tokens, vector_size, window, sg=True):
    if sg:
        model = gensim.models.Word2Vec(text_tokens, min_count=1, vector_size=vector_size, window=window, sg=1)
    else:
        model = gensim.models.Word2Vec(text_tokens, min_count=1, vector_size=vector_size, window=window)
    return model


def get_all_words_from_model(model):
    return sorted(word for word in model.wv.index_to_key)


def get_words_and_vector_array_in_order_from_model(model):
    words = get_all_words_from_model(model)
    vecs = [model.wv[w] for w in words]
    arr = np.array(vecs)
    return words, arr


def combine_model_embeddings(models):
    # linear combination of the embeddings
    # start with uniform weighting for simplicity
    words = None
    shape = None
    arrs = []
    for model in models:
        these_words, this_arr = get_words_and_vector_array_in_order_from_model(model)
        if words is None:
            words = these_words
        else:
            assert these_words == words, "mismatch in words between models"
        if shape is None:
            shape = this_arr.shape
        else:
            assert this_arr.shape == shape, "mismatch in array shapes between models"
        arrs.append(this_arr)
    arr = sum(arrs) / len(arrs)
    assert arr.shape == shape
    return words, arr


def get_word_to_vector_dict(words, vector_arr):
    assert len(words) == vector_arr.shape[0]
    return {words[i]: vector_arr[i, :] for i in range(len(words))}


def get_nearest_neighbors(test_words, test_vecs, word_to_vector_dict):
    # trying to interpret a novel vector that is output by a neural net
    # what are the nearby meanings in the semantic space, and how far away are they?
    k = 5
    all_words = sorted(word_to_vector_dict.keys())
    all_vecs = [word_to_vector_dict[w] for w in all_words]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(all_vecs)
    distances, indices = nbrs.kneighbors(test_vecs)
    d = {}
    for i in range(len(test_words)):
        w = test_words[i]
        these_distances = distances[i]
        these_indices = indices[i]
        neighbor_words = [all_words[i] for i in these_indices]
        d[w] = dict(zip(neighbor_words, these_distances))
    return d


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_word_to_vector_dict_combined_model(text_tokens, vector_size, min_window, max_window_incl, sg=True):
    models = []

    for window in range(min_window, max_window_incl+1, 2):
        model = get_model(text_tokens, vector_size, window, sg)
        models.append(model)
    
    words, vecs = combine_model_embeddings(models)
    return get_word_to_vector_dict(words, vecs)



if __name__ == "__main__":
    # TODO why are the words all so similar with larger corpus?
    # - need to understand what Word2Vec is doing here

    eng_to_lang = rst.get_translation_dict()
    eng_text_tokens = get_eng_text_tokenized()
    # lang_text_tokens = get_lang_text_tokenized()

    words = get_all_words_from_text_tokens(eng_text_tokens)
    # lang_words = get_all_words_from_text_tokens(lang_text_tokens)

    wv = get_word_to_vector_dict_combined_model(eng_text_tokens, vector_size=100, min_window=3, max_window_incl=13, sg=True)

    eng_test_words = random.sample(words, 5)
    # lang_test_words = rst.translate_word_glosses(eng_test_words, eng_to_lang)

    # see what they are doing when they get really similar to each other
    for w in eng_test_words:
        v = wv[w]
        print(f"{w} has vector {v} of shape {v.shape}")

    for w1, w2 in itertools.combinations(eng_test_words, 2):
        v1 = wv[w1]
        v2 = wv[w2]
        sim = cosine_similarity(v1, v2)
        print(f"Cosine similarity between '{w1}' and '{w2}': ", sim)
    