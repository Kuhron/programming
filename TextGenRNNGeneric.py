import sys
from textgenrnn import textgenrnn


if __name__ == "__main__":
    fp = sys.argv[1]
    with open(fp) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]

    textgen = textgenrnn()
    # initialize new model on these texts
    textgen.train_on_texts(lines, num_epochs=1, new_model=True)

    while True:
        epochs = 20
        textgen.train_on_texts(lines, num_epochs=epochs, gen_epochs=epochs+1)
        # don't use new_model here because want to keep older knowledge
        # use gen_epochs greater than num_epochs to prevent it from generating every single epoch
        textgen.generate_samples(n=3, temperatures=[0.25, 0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

