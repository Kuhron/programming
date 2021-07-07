import sys
import os
from textgenrnn import textgenrnn
import numpy as np


def find_ideal_temperature_range(model, texts, starting_temp, step):
    # assumes that copying_proportion is monotonic in temperature
    assert starting_temp > 0
    assert step > 0
    too_many_copies = 0.5
    not_enough_copies = 0

    good_temps = []
    seen_temps = set()
    is_too_uncreative = lambda p: p >= too_many_copies
    is_too_creative = lambda p: p <= not_enough_copies
    is_good = lambda p: (not is_too_uncreative(p)) and (not is_too_creative(p))
    get_p = lambda temp: get_copying_proportion(temp, model, texts)  # shorthand

    t = starting_temp
    p = get_p(t)
    if is_too_uncreative(p):
        direction = +1
    elif is_too_creative(p):
        direction = -1
    else:
        direction = 0  # starting_temp is in good range

    if direction == 0:
        # procedure: go in both directions until reach a bad temp in BOTH directions
        t = starting_temp
        good_temps.append(t)
        # first go up
        while True:
            t += step
            p = get_p(t)
            if is_good(p):
                good_temps.append(t)
            else:
                break
        # then go down
        # reset t to starting_temp so we're not descending back from the high point we just got to
        t = starting_temp
        while True:
            t -= step
            if t <= 0:
                break
            p = get_p(t)
            if is_good(p):
                good_temps.append(t)
            else:
                break

    else:
        # procedure: go in direction until found a good temp, then keep going until found a bad temp, then stop
        found_first_good_temp = False
        while True:
            t += direction * step
            if t <= 0:
                break
            p = get_p(t)
            if is_good(p):
                good_temps.append(t)
                if not found_first_good_temp:
                    found_first_good_temp = True
            else:
                # got a bad temp, if we have already seen the first good temp then that means we are past the good temp range
                if found_first_good_temp:
                    break
    
    return good_temps


def get_copying_proportion(temperature, model, texts):
    if temperature <= 0:
        raise ValueError("non-positive temperature")
    n_samples = 10  # can reduce this if it takes too long to generate
    samples = model.generate(n=n_samples, return_as_list=True, temperature=[temperature])
    assert len(samples) == n_samples  # because idk what the top_n=3 kwarg does in the function call
    n_copies = sum(x in texts for x in samples)
    res = n_copies / n_samples
    print(f"temperature {temperature} has copying proportion of {res}")
    return res


def generate_for_temps_clean(n_samples_per_temp, temps, model, texts):
    for temp in temps:
        samples = model.generate(n=n_samples_per_temp, return_as_list=True, temperature=[temp])
        for sample in samples:
            if sample not in texts:
                print(f"{temp:.2f} : {sample}")


if __name__ == "__main__":
    fps = sys.argv[1:]
    texts = []
    for fp in fps:
        with open(fp) as f:
            lines = f.readlines()
        texts += [l.strip() for l in lines]
    texts = [x for x in texts if x != ""]
    counts = {}
    for x in texts:
        if x not in counts:
            counts[x] = 0
        counts[x] += 1
    duplicates = sorted(x for x,count in counts.items() if count > 1)
    if len(duplicates) > 0:
        for x in duplicates:
            print(f"the following is a duplicate: {x}\n")
        print("exiting due to duplicates")
        sys.exit()

    model_dir = "/home/wesley/programming/NeuralNetFiles/TextGenRNNGeneric/"
    existing_models = [x for x in os.listdir(model_dir) if x.endswith(".hdf5") and not x.endswith("_weights.hdf5")]
    print("existing models:")
    for i, x in enumerate(existing_models):
        print(f"{i}. {x}")
    model_index = input("\nPlease select the model index you want to load, or press enter to create a new model: ")
    try:
        model_index = int(model_index)
        model_filename = existing_models[model_index]
        assert model_filename.endswith(".hdf5")
        model_name = model_filename.replace(".hdf5","")
        print(f"You selected this model: {model_filename}")
        model_fp = os.path.join(model_dir, model_filename)
        loading_existing_model = True
    except (ValueError, IndexError):
        # either a non-int was typed, or a non-existing index
        print(f"No model for your input: {repr(model_index)}. Creating new model.")
        model_name = input("Name for new model: ")
        assert not any(model_name.endswith(x) for x in ["_vocab", "_config", "_weights"])
        model_filename = f"{model_name}.hdf5"
        model_fp = os.path.join(model_dir, model_filename)
        loading_existing_model = False
        if model_filename in existing_models:
            raise ValueError(f"model already exists: {model_filename}")

    # need to tell it various files it uses for metadata, otherwise you'll get stuff like weight array shape not matching when you try loading an existing model
    vocab_fp = os.path.join(model_dir, f"{model_name}_vocab.json")
    config_fp = os.path.join(model_dir, f"{model_name}_config.json")
    weights_fp = os.path.join(model_dir, f"{model_name}_weights.hdf5")

    if loading_existing_model:
        # want ValueErrors here to be raised
        textgen = textgenrnn(config_path=config_fp, weights_path=weights_fp, vocab_path=vocab_fp)
        textgen.load(model_fp)
        print("loaded model")
    else:
        textgen = textgenrnn()
        # initialize new model on these texts, don't use whatever existing English knowledge it seems to come pre-loaded with
        textgen.train_on_texts(texts, num_epochs=1, new_model=True)
        textgen.save(model_fp)
        # will have to move its configs, etc. to the model-specific path
        os.rename("./textgenrnn_config.json", config_fp)
        os.rename("./textgenrnn_vocab.json", vocab_fp)
        os.rename("./textgenrnn_weights.hdf5", weights_fp)

        # now load from the correct config fps
        textgen = textgenrnn(config_path=config_fp, weights_path=weights_fp, vocab_path=vocab_fp)
        textgen.load(model_fp)

    good_temps = None
    while True:
        epochs = 20
        n_samples_per_temp = 10
        if epochs > 0:
            textgen.train_on_texts(texts, num_epochs=epochs, gen_epochs=epochs+1)
        # don't use new_model here because want to keep older knowledge
        # use gen_epochs greater than num_epochs to prevent it from generating every single epoch
        if good_temps is None or good_temps == []:
            starting_temp = 1.0
            step = 0.25
        else:
            starting_temp = np.mean(good_temps)
            step = np.std(good_temps) / 2 if len(good_temps) > 1 else 0.25  # prevent step = 0
        good_temps = find_ideal_temperature_range(textgen, texts, starting_temp, step)
        print(f"good temperature range: {good_temps}")
        if good_temps != []:
            generate_for_temps_clean(n_samples_per_temp, good_temps, textgen, texts)

        textgen.save(model_fp)
        print("saved model")
