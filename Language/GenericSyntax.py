# define a way to input sentences by a morphosyntactic structure
# which can then be subject to movement, inflection, etc. depending on the language being translated to
# use this input format rather than English
# the input language should make all distinctions you would care to make; be specific with semantics and relations
# output will be based on a specific language object's parameters, vocabulary, allomorphy, etc.
# output will always be text but can represent spoken language, sign language, music, or whatever else
# in the case of music, can use MusicParser.py to create sound file from this
# will dramatically improve conlanging ability


import nltk

basic_indonesian_grammar = nltk.CFG.fromstring("""
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

parser = nltk.ChartParser(basic_indonesian_grammar)
# parser = nltk.ChartParser(grammar, trace=1)

sents = [
    # "saya suka kucing merah itu",
    "orang ini bilang bahwa mobil kuning kita adalah dalam rumah besar hijau dia di Jakarta",
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
        break


# now the idea is to have a new grammar/vocab that the tree can be translated into

basic_test_grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> N | VP "olan" NP | Pro | PropN | PP "olan" NP | AP NP | NP Det | "ki" S
VP -> NP V NP | PP VP | AP
AP -> A | A AP
A -> "buyuk" | "kucuk" | "kirmizi" | "sari" | "yesil" | "mavi"
Pro -> "ben" | "sen" | "o" | "biz" | "biz" | "siz" | "onlar"
Det -> "bu" | "su" | Pro
N -> "insan" | "kedi" | "araba" | "ev"
PropN -> "Jakarta" | "Indonesia"
V -> "gore" | "begene" | "yiye" | "gide" | "isteye" | "soyleye"
PP -> NP P
P -> "onda" | "icinde" | "ile"
""")

dict_indonesian_to_test = {
    "besar": "buyuk",
    "kecil": "kucuk",
    "merah": "kirmizi",
    "kuning": "sari",
    "hijau": "yesil",
    "biru": "mavi",
    "saya": "ben",
    "kamu": "sen",
    "dia": "o",
    "kita": "biz",
    "kami": "biz",
    "anda": "siz",
    "mereka": "onlar",
    "ini": "bu",
    "itu": "su",
    "orang": "insan",
    "kucing": "kedi",
    "mobil": "araba",
    "rumah": "ev",
    "lihat": "gore",
    "suka": "begene",
    "makan": "yiye",
    "jalan": "gide",
    "perlu": "isteye",
    "bilang": "soyleye",
    "di": "onda",
    "dalam": "icinde",
    "dengan": "ile",
}
# if item not in dict (e.g. Jakarta), leave it the same

# two steps:
# - translate words in individual nodes
# - rearrange structure, including replacing/inserting/deleting function words which are part of the syntax, e.g. bahwa/ki; adalah/[nothing]; [nothing]/olan
# it seems these are pretty independent, as long as you don't go trying to translate over the function words if you rearrange first
# i.e., only translate words which are "in a node of their own"
# note that "bahwa" in the parsed tree does not have a POS; it is just by itself as part of the constituent above

