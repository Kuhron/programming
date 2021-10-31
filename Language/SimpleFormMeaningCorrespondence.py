# for the product-oriented schema extractor


import random
import numpy as np

from SimpleSoundSystem import SoundVectorSeries
from SimpleMeaningSystem import MeaningVector


if __name__ == "__main__":
    for i in range(100):
        w = SoundVectorSeries.random()
        m = MeaningVector.random()
        # print(f"{w.string} means {m.gloss}")
        print(f"{m.bit_string} : {w.bit_string}")
