from xml.etree import ElementTree as ET
import random

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
    for text in texts:
        print("----")
        print(text)
        print(text.get_contents())
        print("----\n")
