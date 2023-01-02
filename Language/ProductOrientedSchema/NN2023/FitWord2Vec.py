# adapted from https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/

import gensim
import re
import itertools


def sent_tokenize(s):
    pattern = "[^.]+\."
    return re.findall(pattern, s)


def word_tokenize(s):
    pattern = "\w+"
    return re.findall(pattern, s)


if __name__ == "__main__":
    with open("sentences.txt") as f:
        s = f.read()

    f = s.replace("\n", " ")

    text_tokens = []
    for sent in sent_tokenize(f):
        sent_tokens = []
        for word in word_tokenize(sent):
            sent_tokens.append(word.lower())
        text_tokens.append(sent_tokens)

    # Create CBOW model
    model1 = gensim.models.Word2Vec(text_tokens, min_count = 1,
                                vector_size = 100, window = 5)
    # Create Skip Gram model
    model2 = gensim.models.Word2Vec(text_tokens, min_count = 1, vector_size = 100,
                                                window = 5, sg = 1)

    test_words = ["mutu", "sapi", "poko"]

    for w1, w2 in itertools.combinations(test_words, 2):
        print(f"Cosine similarity between '{w1}' and '{w2}' - CBOW : ", model1.wv.similarity(w1, w2))
        print(f"Cosine similarity between '{w1}' and '{w2}' - Skip Gram : ", model2.wv.similarity(w1, w2))