from xml.etree import ElementTree as ET
import random

# FlexPy-specific classes
from RtDict import RtDict
from Text import Text


def get_elements_by_owner_guid(rt_dict, owner_guid):
    return rt_dict.get_by_owner_guid(owner_guid)


def run_flashcards(project_name, rt_dict):
    # asks user random lexicon entries as a vocabulary quiz
    lex_entries = rt_dict["LexEntry"]
    for lex in lex_entries.values():
        lexeme_form_tag = lex.find("LexemeForm")
        lexeme_form_guid = lexeme_form_tag.find("objsur").attrib["guid"]
        senses_tag = lex.find("Senses")
        if senses_tag is None:
            continue
        senses_guid = senses_tag.find("objsur").attrib["guid"]
        form_classes = ["MoStemAllomorph", "MoAffixAllomorph", "WfiWordform"]
        form_found = False
        for fcl in form_classes:
            try:
                lexeme_form_rt = rt_dict["MoStemAllomorph"][lexeme_form_guid]
                lexeme_form = lexeme_form_rt.find("Form").find("AUni").text
                form_found = True
            except KeyError:
                continue
    
        if not form_found:
            print("form not found with guid {}".format(lexeme_form_guid))
            input("press enter to move on to next rt tag")
            continue
    
        sense_rt = rt_dict["LexSense"][senses_guid]
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
    rt_dict = RtDict.from_root(root)

    texts = get_texts(rt_dict)
    text = random.choice(texts)
    print(text)

    run_flashcards(project_name, rt_dict)
    
