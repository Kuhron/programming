# testing Kris's library
from corpus_toolkit import corpus_tools as ct

text1 = "cat cat dog cat elephant"
text2 = "the of and a to in is that it was"
text3 = "a a a a a aa a a a aa aaaaa"
texts = [text1, text2, text3]
tokenized = ct.tokenize(texts)
freq = ct.frequency(tokenized)
ct.head(freq, hits=5)
# frequency function iterates over "tokenized texts" first, then over ttokens in those texts, so input must be an iterable of iterables
# see source at https://github.com/kristopherkyle/corpus_toolkit/blob/b5f0eba13dee60a0b56a25c5f3f900fe7c8c8cb4/build/lib/corpus_toolkit/corpus_tools.py

# what if I include capitals and punctuation
print("----")
text4 = "Cat? Dog! A man, a plan, a canal, Panama! A dog, a panic in a pagoda. Most most most most most most most most most, most? Most! MOST... most?!?! most: Most most most most most most, (most) [most]."
texts.append(text4)
tokenized = ct.tokenize(texts)
freq = ct.frequency(tokenized)
ct.head(freq, hits=10)


print("----")
corpora_dir = "/home/wesley/Desktop/UOregon Work/CorpusLinguistics/corpora/"
text_files = ["dracula.txt", "wuthering_heights.txt"]
texts = []
for text_file in text_files:
    fp = corpora_dir + text_file
    with open(fp) as f:
        contents = f.read()
        texts.append(contents)
tokenized = ct.tokenize(texts)
freq = ct.frequency(tokenized)
ct.head(freq, hits=10)

print("----")
words_to_collocate = ["the"]
for word in words_to_collocate:
    collocates = ct.collocator(tokenized, word, stat="MI")
    print("collocations for word {}:".format(word))
    print("collocates =", collocates)
    ct.head(collocates, hits=10)
