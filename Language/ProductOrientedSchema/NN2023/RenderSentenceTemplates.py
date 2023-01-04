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
    eng_to_lang = {}
    for eng, lang in pairs:
        assert eng not in eng_to_lang, f"{eng} already has a translation"
        eng_to_lang[eng] = lang
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


def translate_sentence_gloss(s, d):
    words = s.split()
    w2 = translate_word_glosses(words, d)
    s2 = " ".join(w2)
    return s2


def translate_word_gloss(w, d):
    # if the morpheme gloss has a dot (.), it needs to be in the dict
    # but if it has a dash, we can ignore the dash and concat the strings
    morphemes = w.split("-")
    w2 = "".join(d[m] for m in morphemes)
    return w2


def translate_word_glosses(words, d):
    return [translate_word_gloss(w, d) for w in words]


def write_sample_files_all_possibilities(templates, translations):
    eng_sample = ""
    lang_sample = ""
    for template in templates:
        # print(f"\ntemplate: {template}")
        eng_renditions = render_template_all_possibilities_eng(template)
        for eng in eng_renditions:
            eng_sample += eng + "\n"
            lang = translate_sentence_gloss(eng, translations)
            lang_sample += lang + "\n"
    write_eng_sample(eng_sample)
    write_lang_sample(lang_sample)


def write_sample_files_random(templates, translations, n_samples):
    eng_lines = []
    lang_lines = []
    for i in range(n_samples):
        eng = render_random_template_eng(templates)
        eng_lines.append(eng)
        lang = translate_sentence_gloss(eng, translations)
        lang_lines.append(lang)
    write_eng_sample(eng_lines)
    write_lang_sample(lang_lines)
    write_glosses_together(eng_lines, lang_lines)


def write_glosses_together(eng_lines, lang_lines):
    with open("sample_glosses_together.txt", "w") as f:
        for eng_line, lang_line in zip(eng_lines, lang_lines):
            f.write(lang_line + "\n" + eng_line + "\n\n")


def write_eng_sample(eng_lines):
    with open("eng_sample.txt", "w") as f:
        for l in eng_lines:
            f.write(l + "\n")
    

def write_lang_sample(lang_lines):
    with open("lang_sample.txt", "w") as f:
        for l in lang_lines:
            f.write(l + "\n")



if __name__ == "__main__":
    templates = get_sentence_template_strings()
    translations = get_translation_dict()
    # write_sample_files_all_possibilities(templates, translations)
    write_sample_files_random(templates, translations, n_samples=100000)
