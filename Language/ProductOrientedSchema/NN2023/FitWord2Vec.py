# adapted from https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

import gensim
import random
import re
import itertools
import numpy as np

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


def convert_words_to_semantic_nn_input(words, text_tokens, sg=True):
    # fit a new Word2Vec model on the given text, then make an array of the words' vectors
    if sg:
        model = gensim.models.Word2Vec(text_tokens, min_count=1, vector_size=100, window=5, sg=1)
    else:
        model = gensim.models.Word2Vec(text_tokens, min_count=1, vector_size=100, window=5)
    n_words = len(words)
    n_cols = model.vector_size
    arr = np.zeros((n_words, n_cols))
    for i, w in enumerate(words):
        vec = model.wv[w]
        arr[i,:] = vec
    return arr



if __name__ == "__main__":
    # TODO why are the words all so similar with larger corpus?
    # - need to understand what Word2Vec is doing here

    eng_to_lang = rst.get_translation_dict()
    eng_text_tokens = get_text_tokenized(eng_sample_fp)
    lang_text_tokens = get_text_tokenized(lang_sample_fp)

    words = get_all_words_from_text_tokens(eng_text_tokens)
    lang_words = get_all_words_from_text_tokens(lang_text_tokens)

    # Create CBOW model
    eng_model1 = gensim.models.Word2Vec(eng_text_tokens, min_count=1, vector_size=100, window=5)
    lang_model1 = gensim.models.Word2Vec(lang_text_tokens, min_count=1, vector_size=100, window=5)

    # Create Skip Gram model
    eng_model2 = gensim.models.Word2Vec(eng_text_tokens, min_count=1, vector_size=100, window=5, sg=1)
    lang_model2 = gensim.models.Word2Vec(lang_text_tokens, min_count=1, vector_size=100, window=5, sg=1)

    eng_test_words = random.sample(words, 5)
    lang_test_words = rst.translate_words(eng_test_words, eng_to_lang)

    # see what they are doing when they get really similar to each other
    for w in eng_test_words:
        v1 = eng_model1.wv[w]
        v2 = eng_model2.wv[w]
        print(f"{w} has CBOW vector {v1} of shape {v1.shape}")
        print(f"{w} has Skip Gram vector {v2} of shape {v2.shape}")
    for w in lang_test_words:
        v1 = lang_model1.wv[w]
        v2 = lang_model2.wv[w]
        print(f"{w} has CBOW vector {v1} of shape {v1.shape}")
        print(f"{w} has Skip Gram vector {v2} of shape {v2.shape}")

    for w1, w2 in itertools.combinations(eng_test_words, 2):
        print(f"Cosine similarity between '{w1}' and '{w2}' - CBOW : ", eng_model1.wv.similarity(w1, w2))
        print(f"Cosine similarity between '{w1}' and '{w2}' - Skip Gram : ", eng_model2.wv.similarity(w1, w2))
    for w1, w2 in itertools.combinations(lang_test_words, 2):
        print(f"Cosine similarity between '{w1}' and '{w2}' - CBOW : ", lang_model1.wv.similarity(w1, w2))
        print(f"Cosine similarity between '{w1}' and '{w2}' - Skip Gram : ", lang_model2.wv.similarity(w1, w2))
    