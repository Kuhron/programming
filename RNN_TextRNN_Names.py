import os
from textgenrnn import textgenrnn

from RNN_GAN_NameGame import get_names


if __name__ == "__main__":
    names_fps = [
        # "/home/wesley/Desktop/Construction/NameMakerRatings.txt",
        "/home/wesley/Desktop/Construction/NameMakerRatings-2.txt",
    ]
    names = get_names(names_fps, repeat_by_rating=False)
    print(f"got {len(names)} names in dataset")
    print(names)

    model_fp = "/home/wesley/programming/NeuralNetFiles/TextGenRNN_Names.hdf5"
    textgen = textgenrnn()
    if os.path.exists(model_fp):
        textgen.load(model_fp)  # I expected this to be a static method, but it's not; must instantiate first
        print("loaded model from file")

    while True:
        textgen.train_on_texts(names, num_epochs=1)
        textgen.generate_samples(n=3, temperatures=[1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])  # additional more creative samples
        textgen.save(model_fp)
        print("---- model saved ----")

