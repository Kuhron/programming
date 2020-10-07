from xml.etree import ElementTree as ET
import random

from Text import Text


def get_rt_dict(root):
    # all <rt> elements in the XML
    # the dictionary is keyed by "class" attribute (e.g. "LexEntry") and then by FLEX's identifier for each object
    rts = root.findall("rt")
    d = {}
    for rt in rts:
        c = rt.attrib["class"]
        if c not in d:
            d[c] = {}
        guid = rt.attrib["guid"]
        d[c][guid] = rt
    return d
 

def run_flashcards(project_name, rt_dict):
    # asks user random lexicon entries as a vocabulary quiz
    lex_entries = d["LexEntry"]
    for lex in lex_entries.values():
        lexeme_form_tag = lex.find("LexemeForm")
        lexeme_form_guid = lexeme_form_tag.find("objsur").attrib["guid"]
        senses_tag = lex.find("Senses")
        senses_guid = senses_tag.find("objsur").attrib["guid"]
        form_classes = ["MoStemAllomorph", "MoAffixAllomorph", "WfiWordform"]
        form_found = False
        for fcl in form_classes:
            try:
                lexeme_form_rt = d["MoStemAllomorph"][lexeme_form_guid]
                lexeme_form = lexeme_form_rt.find("Form").find("AUni").text
                form_found = True
            except KeyError:
                continue
    
        if not form_found:
            print("form not found with guid {}".format(lexeme_form_guid))
            input("press enter to move on to next rt tag")
            continue
    
        sense_rt = d["LexSense"][senses_guid]
        sense = sense_rt.find("Gloss").find("AUni").text
    
        print("\n--- new item ---")
        lang_first = random.random() < 0.7
        if lang_first:
            print("{}: {}".format(project_name, lexeme_form))
            input("English?: ")
            print("answer: {}".format(sense))
        else:
            print("English: {}".format(sense))
            input("{}?: ".format(project_name))
            print("answer: {}".format(lexeme_form))
        input("press enter to continue")
        # print(lexeme_form, sense)
        # input("press2")


def get_texts(rt_dict):
    text_elements = rt_dict["Text"]
    texts = []
    for guid, rt in text_elements.items():
        text = Text(guid, rt)
        texts.append(text)
    print("there are {} texts with contents".format(sum(x.has_contents() for x in texts)))
    return texts



if __name__ == "__main__":
    project_name = "Bongu"
    flex_dir = "/home/wesley/.local/share/fieldworks/Projects/{}/".format(project_name)
    fp = flex_dir + "{}.fwdata".format(project_name)
    print("processing project {} at {}".format(project_name, fp))

    # in vim, type gg=G in normal mode (no colon) to indent xml file
    # this works because I added the following in ~/.vimrc
    # :set equalprg=xmllint\ --format\ -
    # and can type ':set nowrap' (no quotes) to stop wrapping the long lines of semantic domain descriptions
    # easier to see the .fwdata structure this way
 
    tree = ET.parse(fp)
    root = tree.getroot()
    rt_dict = get_rt_dict(root)

    texts = get_texts(rt_dict)
    text = random.choice(texts)
    print(text)
    
