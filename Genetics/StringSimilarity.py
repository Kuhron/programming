import os
import random

from Genome import Genome
# from Levenshtein import levenshtein  # this one was my Levenshtein.py in programming dir, but then I removed that dir from pythonpath (since having it there seems like a hack, I don't want imports trying to look there if I am elsewhere)
import Levenshtein as lev  # this is the python-Levenshtein library from PyPI


def get_genome_fps():
    data_dir = "ncbi_dataset/data/individual_genomes/"
    return [os.path.join(data_dir, x) for x in os.listdir(data_dir)]


def get_random_fp():
    return random.choice(get_genome_fps())


def get_n_genomes(n):
    return random.sample(get_genome_fps(), n)




if __name__ == "__main__":
    fp1, fp2 = get_n_genomes(2)
    print("fp1: {}\nfp2: {}".format(fp1, fp2))
    g1 = Genome.from_nih_file(fp1)
    g2 = Genome.from_nih_file(fp2)
    distance = lev.distance(g1.string, g2.string)
    print("distance = {}".format(distance))
