# messing around with colexifications using CLICS database
# citation: Rzymski, Tresoldi et al. 2019. The Database of Cross-Linguistic Colexifications, reproducible analysis of cross- linguistic polysemies. DOI: doi.org/10.17613/5awv-6w15

# instructions for downloading/compiling the data: https://github.com/clics/clics3
# I have already done this on 2021-07-09 but am not committing all that stuff to my repo (it's in other repos out there as listed in datasets.txt)
# to check what datasets you have, run `clics datasets` in a terminal

# want to be able to mess with CLICS data my own way, not just run their code to create the GML graph
# e.g. take some subset of languages and make a semantic map out of just those ones
# e.g. perform random walk on concepts to create new semantic map for conlang


import os
import csv
import random
import itertools
import math
import scipy.stats
# import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

from BinomialObservation import BinomialObservation  # might need sys.path to look at parent dir

class Concept:
    def __init__(self, concept_id, concept_gloss, source_glosses):
        self.concept_id = concept_id
        self.concept_gloss = concept_gloss
        self.source_glosses = source_glosses  # list of all strings used as gloss for this same concept

    def __repr__(self):
        return f"<Concept id=\"{self.concept_id}\" gloss=\"{self.concept_gloss}\">"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if type(other) is not Concept:
            return NotImplemented
        if self.concept_id == other.concept_id:
            assert self.concept_gloss == other.concept_gloss, f"two concepts with same id but different gloss:\n{self}\n{other}"
            return True
        else:
            return False


class ConceptEncoding:
    def __init__(self, language, form, concept):
        self.language = language
        self.form = form
        assert type(concept) is Concept
        self.concept = concept

    def __repr__(self):
        return f"<ConceptEncoding: language \"{self.language}\" encodes {self.concept} as \"{self.form}\">"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if type(other) is not ConceptEncoding:
            return NotImplemented
        return self.language == other.language and self.form == other.form and self.concept == other.concept


def get_subdirs_with_data():
    directory = "/home/wesley/programming/Language/src/"
    subdirs = [f for f in os.scandir(directory) if f.is_dir()]
    database_names = []
    file_dirs = []
    for subdir in subdirs:
        database_names.append(subdir.name)
        if subdir.name == "lexibank-hantganbangime":
            # for some reason this one has an extra src/ and a subdir with the same name EXCEPT hyphen is replace with underscore. WHY??
            file_dir = "src/lexibank_hantganbangime/cldf"
        else:
            file_dir = "cldf"
        file_dir_path = os.path.join(directory, subdir, file_dir)
        file_dirs.append(file_dir_path)
    return file_dirs, database_names


def get_filepaths_for_filename(default_filename, exceptions_db_to_filename=None):
    # some dbs are stupid and made their files have a different name, why?! especially freaking pylexirumah doesn't follow the rules
    # exceptions_db_to_filename is meant to account for this
    file_dirs, database_names = get_subdirs_with_data()
    fps = []
    for file_dir, db_name in zip(file_dirs, database_names):
        if exceptions_db_to_filename is not None and db_name in exceptions_db_to_filename:
            filename = exceptions_db_to_filename[db_name]
        else:
            filename = default_filename

        fp = os.path.join(file_dir, filename)
        if os.path.exists(fp):
            fps.append(fp)
        else:
            print(f"Warning: forms file does not exist: {fp}")
    return fps, database_names



def get_forms_filepaths():
    return get_filepaths_for_filename("forms.csv")


def get_languages_filepaths():
    return get_filepaths_for_filename("languages.csv", {"pylexirumah": "lects.csv"})


def get_parameters_filepaths():
    return get_filepaths_for_filename("parameters.csv", {"pylexirumah": "concepts.csv"})


def get_raw_concepts():
    concepts_by_database = get_raw_concepts_by_database()
    res = []
    for db, concepts_by_id in concepts_by_database.items():
        for idx, concept in concepts_by_id.items():
            assert type(concept) is dict
            res.append(concept)
    return res


def get_dict_rows(fp):
    with open(fp) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def get_raw_concept_from_row(row, db_name):
    if db_name == "pylexirumah":
        # no Concepticon_Gloss here
        # the concept's name is in the ID field instead of Name, will use this as gloss if there is no Concepticon_Gloss for the associated Concepticon_ID
        source_gloss = row["ID"]
        d = {"Concepticon_ID": row["Concepticon_ID"], "Source_Gloss": source_gloss}
    else:
        # in the parameters.csv, it's ID not Parameter_ID
        source_gloss = row["Name"]
        d = {"Concepticon_ID": row["Concepticon_ID"], "Concepticon_Gloss": row["Concepticon_Gloss"], "Source_Gloss": source_gloss}
    return d


def get_raw_concepts_by_database():
    parameters_fps, db_names = get_parameters_filepaths()
    concepts = {}
    for parameters_fp, db_name in zip(parameters_fps, db_names):
        # print(f"getting concepts from database {db_name}")
        concepts[db_name] = {}
        rows = get_dict_rows(parameters_fp)

        for row in rows:
            raw_concept = get_raw_concept_from_row(row, db_name)
            assert row["ID"] not in concepts[db_name], f"duplicate ID: {row['ID']}"
            concepts[db_name][row["ID"]] = raw_concept

    return concepts


def get_rows_from_fp(fp, database_name):
    rows = get_dict_rows(fp)
    for row in rows:
        assert "fp" not in row
        row["fp"] = fp  # store which file this record came from
        assert "database_name" not in row
        row["database_name"] = database_name
    return rows


def get_rows_from_fps(fps, database_names):
    rows = []
    all_keys = set()
    keys_in_all_fps = None
    for fp, db_name in zip(fps, database_names):
        fp_rows = get_rows_from_fp(fp, db_name)
        fp_keys = set(fp_rows[0].keys())
        all_keys |= fp_keys
        if keys_in_all_fps is None:
            keys_in_all_fps = fp_keys
        else:
            keys_in_all_fps &= fp_keys
            # don't want to do this with initializing it as empty set because then it will just stay empty
        rows += fp_rows

    # some of them don't have the same keys, leave those keys out (don't set them to some default)
    # so that KeyError is raised if you try to use one
    # print(f"all keys:\n{sorted(all_keys)}\nkeys in all files:\n{sorted(keys_in_all_fps)}")
    return rows, all_keys, keys_in_all_fps


def print_keys_of_rows(rows, keys):
    max_key_len = max(len(k) for k in keys)
    keys = sorted(keys)
    for i, row in enumerate(rows):
        print(f"row {i}")
        for k in keys:
            val = row.get(k)
            print(f"- {k.ljust(max_key_len+1)}: {val}")
        print()


def show_key_statistics(rows, keys):
    # for each key, show how many rows have it (and proportion) and some examples of what is in it
    for k in sorted(keys):
        rows_with_key = [row for row in rows if k in row and row[k] != ""]
        print(f"key {k} is in {len(rows_with_key)} rows out of {len(rows)} ({100*len(rows_with_key)/len(rows):.2f}%). Examples of its values:")
        sample_rows = random.sample(rows_with_key, min(5, len(rows_with_key)))
        for row in sample_rows:
            print(f"{k} : {row[k]}")
        print()


def construct_concepticon_id_gloss_correspondence(raw_concepts):
    gloss_to_id = {}
    id_to_gloss = {}
    for rc in raw_concepts:
        # some have ID with no gloss, some have gloss with no ID
        # some have both, which is great
        # some have neither, in which case we consider it not to be a part of the concepticon
        # (but note that the underlying data has mistakes, so something may actually be part of the concepticon when it shouldn't be,
        # or vice versa, or it may be labeled as the wrong concept, e.g. example of bark(dog) labeled as bark(tree))

        if has_id(rc) and has_gloss(rc):
            concept_id = rc["Concepticon_ID"]
            concept_gloss = rc["Concepticon_Gloss"]
            if concept_id in id_to_gloss:
                existing_gloss = id_to_gloss[concept_id]
                assert concept_gloss == existing_gloss, f"gloss conflict for id {concept_id}: {concept_gloss} != {existing_gloss}"
            else:
                id_to_gloss[concept_id] = concept_gloss
            if concept_gloss in gloss_to_id:
                existing_id = gloss_to_id[concept_gloss]
                assert concept_id == existing_id, f"id conflict for gloss {concept_gloss}: {concept_id} != {existing_id}"
            else:
                gloss_to_id[concept_gloss] = concept_id

    assert len(gloss_to_id) == len(id_to_gloss)
    for k,v in gloss_to_id.items():
        assert id_to_gloss[v] == k
    for k,v in id_to_gloss.items():
        assert gloss_to_id[v] == k
    return id_to_gloss, gloss_to_id


def has_id(raw_concept):
    return "Concepticon_ID" in raw_concept and raw_concept["Concepticon_ID"] != ""


def has_gloss(raw_concept):
    return "Concepticon_Gloss" in raw_concept and raw_concept["Concepticon_Gloss"] != ""


def has_source_gloss(raw_concept):
    rc = raw_concept
    return "Source_Gloss" in rc and rc["Source_Gloss"] != ""


def get_source_gloss(raw_concept):
    rc = raw_concept
    return rc["Source_Gloss"]


def construct_concept_objects(raw_concepts, id_to_gloss, gloss_to_id):
    expected_fields = ["Concepticon_ID", "Concepticon_Gloss", "Source_Gloss"]
    for c in raw_concepts:
        assert all(k in expected_fields for k in c.keys()), f"concept has unexpected field: {c}"

    for rc in raw_concepts:
        rc["has_id"] = has_id(rc)
        rc["has_gloss"] = has_gloss(rc)

    no_id_no_gloss = [rc for rc in raw_concepts if not rc["has_id"] and not rc["has_gloss"]]
    no_id_yes_gloss = [rc for rc in raw_concepts if not rc["has_id"] and rc["has_gloss"]]
    yes_id_no_gloss = [rc for rc in raw_concepts if rc["has_id"] and not rc["has_gloss"]]
    yes_id_yes_gloss = [rc for rc in raw_concepts if rc["has_id"] and rc["has_gloss"]]

    # detect impostors: concepts with a Concepticon_Gloss but no Concepticon_ID, and for whose gloss there is no Concepticon_ID,
    # i.e. the concept is not actually in the concepticon, or if it is, we don't know the ID
    glosses_of_extra_concepts = set()
    for rc in no_id_yes_gloss:
        # again don't prematurely optimize this
        gloss = rc["Concepticon_Gloss"]
        assert gloss != "", "shouldn't have passed has_gloss check"
        # now want to know if any other concept links this gloss to some id in concepticon
        shared_gloss_rcs_with_id = [rc for rc in yes_id_yes_gloss if rc["Concepticon_Gloss"] == gloss]
        if len(shared_gloss_rcs_with_id) == 0:
            # true impostor, there is no ID matching this Concepticon_Gloss
            glosses_of_extra_concepts.add(gloss)
        else:
            ids = set(rc["Concepticon_ID"] for rc in shared_gloss_rcs_with_id)
            assert len(ids) == 1, f"more than one Concepticon_ID found for the same Concepticon_Gloss: {ids}"
            concept_id = list(ids)[0]
            # add it to the correspondence
            id_to_gloss[concept_id] = gloss
            gloss_to_id[gloss] = concept_id

    # detect orphans: concepts with a Concepticon_ID but no Concepticon_Gloss anywhere
    orphan_ids = set()
    for rc in yes_id_no_gloss:
        concept_id = rc["Concepticon_ID"]
        assert concept_id != "", "shouldn't have passed has_id check"
        # now want to know if this id is linked to a gloss somewhere else
        shared_id_rcs_with_gloss = [rc for rc in yes_id_yes_gloss if rc["Concepticon_ID"] == concept_id]
        if len(shared_id_rcs_with_gloss) == 0:
            # true orphan, there is no Concepticon_Gloss for this ID anywhere
            orphan_ids.add(concept_id)
            if has_source_gloss(rc):
                gloss = get_source_gloss(rc)
                # print(f"orphan id {concept_id} has source gloss {gloss}")
                id_to_gloss[concept_id] = gloss
                gloss_to_id[gloss] = concept_id
            else:
                raise Exception(f"orphan id {concept_id} has no source gloss")
        else:
            glosses = set(rc["Concepticon_Gloss"] for rc in shared_id_rcs_with_gloss)
            assert len(glosses) == 1, f"more than one Concepticon_Gloss found for the same Concepticon_ID: {glosses}"
            gloss = list(glosses)[0]
            # add it to the correspondence
            id_to_gloss[concept_id] = gloss
            gloss_to_id[gloss] = concept_id
    # the orphan IDs are found in:
    # lexibank-naganorgyalrongic/cldf/parameters.csv
    # pylexirumah/cldf/concepts.csv

    # the ones with no id or gloss should be treated each as their own concept, based on the name/ID(for pylexirumah) field
    for rc in no_id_no_gloss:
        gloss = get_source_gloss(rc)
        glosses_of_extra_concepts.add(gloss)

    glosses_of_extra_concepts = sorted(glosses_of_extra_concepts)
    # print("glosses_of_extra_concepts:", glosses_of_extra_concepts)
    for i, g in enumerate(glosses_of_extra_concepts):
        new_id = f"EC{i+1}"  # EC stands for EXTRACONCEPT
        assert new_id not in id_to_gloss, new_id
        assert g not in gloss_to_id, g
        id_to_gloss[new_id] = g
        gloss_to_id[g] = new_id

    # now make correspondence from ID to Concept objects, and then go fill in all the glosses from the raw concepts
    id_to_concept = {}
    for concept_id in id_to_gloss:
        concept_gloss = id_to_gloss[concept_id]
        c = Concept(concept_id, concept_gloss, [])
        id_to_concept[concept_id] = c

    # now go through the raw concepts and add the source gloss to the correct Concept object
    for rc in raw_concepts:
        if rc["has_id"]:
            concept_id = rc["Concepticon_ID"]
            c = id_to_concept[concept_id]
        elif rc["has_gloss"]:
            concept_gloss = rc["Concepticon_Gloss"]
            # some of the ID-less ones have a gloss which is in the concepticon, but some have one which is not (a concepticon-impostor, if you will)
            try:
                concept_id = gloss_to_id[concept_gloss]
            except KeyError:
                # if it's an impostor, there is no such concepticon gloss in the real concepticon, so this is an extra concept
                # but its gloss should still already be in the extraconcepts list
                raise Exception(f"impostor: {rc} should have been added to id-gloss correspondence but it wasn't")
            c = id_to_concept[concept_id]
        else:
            concept_gloss = get_source_gloss(rc)
            concept_id = gloss_to_id[concept_gloss]
            c = id_to_concept[concept_id]
        c.source_glosses.append(concept_gloss)
 
    return id_to_concept, id_to_gloss, gloss_to_id


def get_language_from_row(row, language_id_to_glottocode_by_database):
    db_name = row["database_name"]
    id_to_glottocode = language_id_to_glottocode_by_database[db_name]
    if "Language_ID" in row:
        language_id = row["Language_ID"]
    elif "Lect_ID" in row:
        language_id = row["Lect_ID"]
    else:
        raise Exception(f"can't get language from row: {row}")

    try:
        glottocode = id_to_glottocode[language_id]
    except KeyError:
        raise KeyError(f"language_id {language_id} not found in\n{id_to_glottocode}")
    assert glottocode != "", f"row {row} has blank language code"
    return glottocode


def get_language_id_to_glottocode_dict_by_database():
    language_fps, db_names = get_languages_filepaths()
    d = {}
    for language_fp, db_name in zip(language_fps, db_names):
        id_to_glottocode = {}
        rows = get_dict_rows(language_fp)
        # we are reading languages.csv right now
        for row in rows:
            language_id = row["ID"]
            glottocode = row["Glottocode"]
            if glottocode == "":
                # failing this, use the ISO 639-3 (occurs in Basque (West) in lexibank-diacl, for instance)
                iso_code = row["ISO639P3code"]
                if iso_code == "":
                    # failing THAT, use the name! (occurs in Proto-Albanian in lexibank-diacl)
                    lang_name = row["Name"]
                    if lang_name == "":
                        raise ValueError(f"database {db_name} has blank Glottocode, ISO code, and name in row {row}")
                    else:
                        code = lang_name
                else:
                    code = iso_code
            else:
                code = glottocode

            id_to_glottocode[language_id] = code

        d[db_name] = id_to_glottocode
    return d


def get_parameter_id_to_concept_by_database(id_to_concept, id_to_gloss, gloss_to_id):
    parameters_fps, db_names = get_parameters_filepaths()
    d = {}
    for parameters_fp, db_name in zip(parameters_fps, db_names):
        this_db_id_to_concept = {}
        rows = get_dict_rows(parameters_fp)
        # we are reading parameters.csv or concepts.csv
        for row in rows:
            raw_concept = get_raw_concept_from_row(row, db_name)
            parameter_id = row["ID"]
            if has_id(raw_concept):
                # has concepticon ID
                concept_id = raw_concept["Concepticon_ID"]
            elif has_gloss(raw_concept):
                concept_gloss = raw_concept["Concepticon_Gloss"]
                concept_id = gloss_to_id[concept_gloss]
            elif has_source_gloss(raw_concept):
                source_gloss = get_source_gloss(raw_concept)
                concept_id = gloss_to_id[source_gloss]
            else:
                raise Exception(f"cannot get concept for parameter raw_concept {raw_concept}")
            concept = id_to_concept[concept_id]
            this_db_id_to_concept[parameter_id] = concept
        d[db_name] = this_db_id_to_concept
    return d


def get_form_from_row(row):
    # could use Value, but I think it sometimes tends to collapse distinctions e.g. removing diacritics, creating spurious colexifications
    return row["Form"]


def get_concept_parameter_id_from_row(row):
    db_name = row["database_name"]
    if db_name == "pylexirumah":
        return row["Concept_ID"]  # seriously dude, why is this not unified
    else:
        return row["Parameter_ID"]  # this will map via parameters.csv to the concepticon


def get_concept_from_row(row, id_to_concept, id_to_gloss, gloss_to_id, parameter_id_to_concept_by_database):
    db_name = row["database_name"]
    parameter_id = get_concept_parameter_id_from_row(row)
    return parameter_id_to_concept_by_database[db_name][parameter_id]


def get_all_concept_encodings(rows, id_to_concept, id_to_gloss, gloss_to_id, language_id_to_glottocode_by_database, parameter_id_to_concept_by_database):
    all_concept_encodings = []
    for row in rows:
        lang = get_language_from_row(row, language_id_to_glottocode_by_database)
        form = get_form_from_row(row)
        concept = get_concept_from_row(row, id_to_concept, id_to_gloss, gloss_to_id, parameter_id_to_concept_by_database)
        # print(f"language {lang} codes {concept} as {form}")
        concept_encoding = ConceptEncoding(lang, form, concept)
        all_concept_encodings.append(concept_encoding)
    return all_concept_encodings


def get_random_colexification(colexification_sets_by_language):
    for i in range(100):
        lang, colexification_set_by_form = random.choice(list(colexification_sets_by_language.items()))
        sets_with_colexifications = [s for s in colexification_set_by_form.values() if len(s) > 1]
        if len(sets_with_colexifications) == 0:
            print(f"language {lang} has no colexifications")
            continue
        chosen_set = random.choice(sets_with_colexifications)
        pair = random.sample(chosen_set, 2)
        return pair
    print("failed to find a colexification")


def get_colexification_sets_by_language(concept_encodings):
    d = {}
    for ce in concept_encodings:
        lang = ce.language
        if lang not in d:
            d[lang] = {}
        form = ce.form
        if ce.form not in d[lang]:
            d[lang][ce.form] = set()
        d[lang][ce.form].add(ce)
    return d


def add_pair_to_symmetric_sparse_matrix(d, pair, value_change=1):
    c0, c1 = pair
    if c0 not in d:
        d[c0] = {c1: 0}
    if c1 not in d:
        d[c1] = {c0: 0}
    if c1 not in d[c0]:
        d[c0][c1] = 0
    if c0 not in d[c1]:
        d[c1][c0] = 0
    # must be symmetric
    d[c0][c1] += 1
    d[c1][c0] += 1
    return d


def validate_symmetric_sparse_matrix(d):
    for c0 in d.keys():
        for c1 in d[c0].keys():
            v = d[c0][c1]
            assert v > 0  # only store non-zero since it will be very sparse
            assert d[c1][c0] == v  # must be symmetric


def get_concept_closeness_matrix(colexification_sets_by_language):
    d = {}
    for language, colexifications_by_form in colexification_sets_by_language.items():
        for form, concept_encodings in colexifications_by_form.items():
            concepts = [ce.concept for ce in concept_encodings]
            pairs = itertools.combinations(concepts, 2)
            for pair in pairs:
                d = add_pair_to_symmetric_sparse_matrix(d, pair)

    validate_symmetric_sparse_matrix(d)
    return d


def get_concept_cooccurrence_count_matrix(colexification_sets_by_language):
    # for each pair of concepts, count how many times they are both in the same language's dataset
    # (used as number of trials for binomial estimation of colexification probability)
    # only include non-zero values, sparse matrix
    # this is really slow, possibly due to the large number of pairs (combinations(n,2) approaches n**2/2)
    d = {}
    lang_i = 0
    n_langs = len(colexification_sets_by_language.keys())
    for language, colexifications_by_form in colexification_sets_by_language.items():
        all_concepts = set()
        for form, concept_encodings in colexifications_by_form.items():
            concepts_encoded = [ce.concept for ce in concept_encodings]
            all_concepts |= set(concepts_encoded)
            # use set because some concepts may be encoded by more than one form
        for pair in itertools.combinations(all_concepts, 2):
            d = add_pair_to_symmetric_sparse_matrix(d, pair)
        print(f"after language {language} (progress {lang_i}/{n_langs}), cooccurrence count now has length {len(d)}")

    validate_symmetric_sparse_matrix(d)
    return d


def summarize_confidence_intervals(binomial_observation):
    b = binomial_observation
    p_hat = b.get_probability_estimator()
    print(f"binomial observation of {b.successes}/{b.trials}, p_hat = {p_hat}")
    confidences = np.arange(0.001, 1.000, 0.001)  # arange is right-excl, confidence of 1 gives wilson of nan
    nci_lows = []
    nci_highs = []
    wci_lows = []
    wci_highs = []
    p_hats = []
    for confidence_level in confidences:
        nci = b.get_normal_approximation_ci(confidence_level)
        wci = b.get_wilson_ci(confidence_level)
        nci_lows.append(nci[0])
        nci_highs.append(nci[1])
        wci_lows.append(wci[0])
        wci_highs.append(wci[1])
        # print(confidence_level, nci, wci)
    plt.plot(confidences, nci_lows, c="g", label="Normal")
    plt.plot(confidences, nci_highs, c="g")
    plt.plot(confidences, wci_lows, c="r")
    plt.plot(confidences, wci_highs, c="r", label="Wilson")
    plt.legend()
    plt.show()

    random_ps = [b.choose_random_possible_probability() for i in range(100)]
    plt.hist(random_ps, bins=25)
    plt.show()



if __name__ == "__main__":
    fps, db_names = get_forms_filepaths()
    rows, all_keys, keys_in_all_fps = get_rows_from_fps(fps, db_names)

    raw_concepts = get_raw_concepts()
    id_to_gloss, gloss_to_id = construct_concepticon_id_gloss_correspondence(raw_concepts)
    id_to_concept, id_to_gloss, gloss_to_id = construct_concept_objects(raw_concepts, id_to_gloss, gloss_to_id)

    language_id_to_glottocode_by_database = get_language_id_to_glottocode_dict_by_database()
    parameter_id_to_concept_by_database = get_parameter_id_to_concept_by_database(id_to_concept, id_to_gloss, gloss_to_id)
    all_concept_encodings = get_all_concept_encodings(rows, id_to_concept, id_to_gloss, gloss_to_id, language_id_to_glottocode_by_database, parameter_id_to_concept_by_database)
    colexification_sets_by_language = get_colexification_sets_by_language(all_concept_encodings)

    concept_closeness_matrix = get_concept_closeness_matrix(colexification_sets_by_language)
    smaller_colexification_sets_by_language = dict(random.sample(colexification_sets_by_language.items(), 5))  # debug, the cooccurrence counting takes forever
    concept_cooccurrence_matrix = get_concept_cooccurrence_count_matrix(smaller_colexification_sets_by_language)

    for i in range(100):
        while True:
            c0 = random.choice(list(concept_cooccurrence_matrix.keys()))
            c1 = random.choice(list(concept_cooccurrence_matrix[c0].keys()))
            successes = concept_closeness_matrix[c0][c1] if c0 in concept_closeness_matrix and c1 in concept_closeness_matrix[c0] else 0
            if successes > 0:
                break
        trials = concept_cooccurrence_matrix[c0][c1]
        b = BinomialObservation(successes, trials)
        print(c0, c1)
        # summarize_confidence_intervals(b)

    # TODO: also keep track of how often a pair of concepts even exists in a language
    # combining that with the number of times it's colexified, can get Wilson confidence interval for binomial probability
    # when rolling whether to colexify something in the random walk, can use the simple binomial estimator,
    # but would be better to use a pdf derived from the Wilson intervals somehow (e.g. get non-zero probability of colexifying something whose colexification has never been seen)

