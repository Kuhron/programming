# define a way to input sentences by a morphosyntactic structure
# which can then be subject to movement, inflection, etc. depending on the language being translated to
# use this input format rather than English
# the input language should make all distinctions you would care to make; be specific with semantics and relations
# output will be based on a specific language object's parameters, vocabulary, allomorphy, etc.
# output will always be text but can represent spoken language, sign language, music, or whatever else
# in the case of music, can use MusicParser.py to create sound file from this
# will dramatically improve conlanging ability


import nltk

grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> N | N "yang" VP | Pro | PropN | NP AP | NP Det | "bahwa" S
VP -> V NP | VP PP | "adalah" AP
AP -> A | PP | A AP
A -> "besar" | "kecil" | "merah" | "kuning" | "hijau" | "biru"
Pro -> "saya" | "kamu" | "dia" | "kita" | "kami" | "anda" | "mereka"
Det -> "ini" | "itu" | Pro
N -> "orang" | "kucing" | "mobil" | "rumah"
PropN -> "Jakarta" | "Indonesia"
V -> "lihat" | "suka" | "makan" | "jalan" | "perlu" | "bilang"
PP -> P NP
P -> "di" | "dalam" | "dengan"
""")

parser = nltk.ChartParser(grammar)
# parser = nltk.ChartParser(grammar, trace=1)

sents = [
    "saya suka kucing merah itu",
    # "orang ini bilang bahwa mobil kuning kita adalah dalam rumah besar hijau dia di Jakarta",
]

for sent in sents:
    words = sent.split()
    print("parsing: {}".format(sent))
    trees = list(parser.parse(words))
    print("found {} readings".format(len(trees)))
    for tree in trees:
        print(tree)
        print(tree[0])
        print(tree[1][0])
        tree.draw()