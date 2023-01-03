import re
import random

from TemplateWordGroups import template_word_groups

template_fp = "sentence_templates.txt"
translation_fp = "word_translations.txt"


def get_sentence_template_strings():
    with open(template_fp) as f:
        lines = f.readlines()
    l2 = [l.strip() for l in lines]
    return [l for l in l2 if len(l) > 0]


def get_translation_dict():
    with open(translation_fp) as f:
        lines = f.readlines()
    pairs = [l.replace("\n","").split(" = ") for l in lines]
    eng_to_lang = dict(pairs)
    return eng_to_lang


def get_first_replacement_slot(template_str):
    pattern = "\{[^\{\}]+\}"
    first_match = re.search(pattern, template_str)
    if first_match is not None:
        first_match = first_match.group()  # get the text from re.match object
    return first_match


def render_template_all_possibilities_eng(template_str):
    slot_str = get_first_replacement_slot(template_str)
    if slot_str is None:
        return [template_str]
    replacements = template_word_groups[slot_str]
    renditions = [template_str.replace(slot_str, replacement) for replacement in replacements]
    final_renditions = []
    for s in renditions:
        final_renditions += render_template_all_possibilities_eng(s)
    return final_renditions


def render_template_random_possibility_eng(template_str):
    slot_str = get_first_replacement_slot(template_str)
    if slot_str is None:
        return template_str
    replacements = template_word_groups[slot_str]
    replacement = random.choice(replacements)
    rendition = template_str.replace(slot_str, replacement)
    return render_template_random_possibility_eng(rendition)


def render_random_template_eng(templates):
    template = random.choice(templates)
    return render_template_random_possibility_eng(template)


def translate_gloss(s, d):
    words = s.split()
    w2 = translate_words(words, d)
    s2 = " ".join(w2)
    return s2


def translate_words(words, d):
    # if the morpheme gloss has a dot (.), it needs to be in the dict
    # but if it has a dash, we can ignore the dash and concat the strings
    w2 = []
    for gloss in words:
        morphemes = gloss.split("-")
        w = "".join(d[m] for m in morphemes)
        w2.append(w)
    return w2


def write_sample_files_all_possibilities(templates, translations):
    eng_sample = ""
    lang_sample = ""
    for template in templates:
        # print(f"\ntemplate: {template}")
        eng_renditions = render_template_all_possibilities_eng(template)
        for eng in eng_renditions:
            eng_sample += eng + "\n"
            lang = translate_gloss(eng, translations)
            lang_sample += lang + "\n"
    write_eng_sample(eng_sample)
    write_lang_sample(lang_sample)


def write_sample_files_random(templates, translations, n_samples):
    eng_sample = ""
    lang_sample = ""
    for i in range(n_samples):
        eng = render_random_template_eng(templates)
        eng_sample += eng + "\n"
        lang = translate_gloss(eng, translations)
        lang_sample += lang + "\n"
    write_eng_sample(eng_sample)
    write_lang_sample(lang_sample)


def write_eng_sample(eng_sample):
    with open("eng_sample.txt", "w") as f:
        f.write(eng_sample)
    

def write_lang_sample(lang_sample):
    with open("lang_sample.txt", "w") as f:
        f.write(lang_sample)



if __name__ == "__main__":
    templates = get_sentence_template_strings()
    translations = get_translation_dict()
    # write_sample_files_all_possibilities(templates, translations)
    write_sample_files_random(templates, translations, n_samples=5000)
