from xml.etree import ElementTree as ET
import random

# Kris's library
from corpus_toolkit import corpus_tools as ct

# FlexPy-specific classes
from FlexPyUtil import get_single_child
from RtDict import RtDict
from Text import Text


def get_elements_by_owner_guid(rt_dict, owner_guid):
    return rt_dict.get_by_owner_guid(owner_guid)


def get_texts(rt_dict):
    text_elements = rt_dict["Text"]
    texts = []
    for guid, rt in text_elements.items():
        text = Text(guid, rt, rt_dict)
        texts.append(text)
    print("there are {} texts with contents".format(sum(x.has_contents() for x in texts)))
    return texts


def get_frequencies_naive(strs):
    assert type(strs) is list
    d = {}
    chars_to_delete = ".,/?!;\'\""
    for s in strs:
        for c in chars_to_delete:
            s = s.replace(c, "")
        words = s.lower().strip().split()
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
    return d


def report_frequencies_naive(strs):
    d = get_frequencies_naive(strs)
    tups = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    top_n = 50
    total_words = sum(d.values())
    print("---- report of top {} most frequent words (naive) out of {} words total ----".format(top_n, total_words))
    for tup in tups[:top_n]:
        print("Word {0} occurs {1} times".format(*tup))



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
 
    # tree = ET.parse(fp)
    # root = tree.getroot()
    rt_dict = RtDict.from_fwdata_file(fp)

    texts = get_texts(rt_dict)
    contents_lst = []
    for text in texts:
        print("----")
        print(text)
        contents = text.get_contents()
        print(contents)
        contents_lst += contents
        print("----\n")

    report_frequencies_naive(contents_lst)


