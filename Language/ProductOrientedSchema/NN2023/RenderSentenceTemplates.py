import re

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


def render_template_all_possibilities_eng(template_str):
    pattern = "\{[^\{\}]+\}"
    first_match = re.search(pattern, template_str)
    if first_match is None:
        return [template_str]
    else:
        first_match = first_match.group()  # get the text
    group_name = first_match.replace("{","").replace("}","")
    replacements = template_word_groups[group_name]
    renditions = [template_str.replace(first_match, replacement) for replacement in replacements]
    final_renditions = []
    for s in renditions:
        final_renditions += render_template_all_possibilities_eng(s)
    return final_renditions


def translate_gloss(s, d):
    words = s.split()
    w2 = [d[w] for w in words]
    s2 = " ".join(w2)
    return s2


if __name__ == "__main__":
    templates = get_sentence_template_strings()
    translations = get_translation_dict()
    for template in templates:
        print(f"\ntemplate: {template}")
        eng_renditions = render_template_all_possibilities_eng(template)
        for eng in eng_renditions:
            lang = translate_gloss(eng, translations)
            print(f"{lang}.")