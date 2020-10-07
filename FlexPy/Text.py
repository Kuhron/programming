from FlexPyUtil import get_single_child

class Text:
    def __init__(self, guid, rt, rt_dict):
        self.guid = guid
        self.rt = rt
        self.rt_dict = rt_dict
        self.name = self.get_name()
        self.contents = self.get_contents()


    def get_name(self):
        try:
            abbreviation = get_single_child(self.rt, "Abbreviation")
            auni = get_single_child(abbreviation, "AUni")
            return auni.text
        except:
            return None


    def get_contents(self):
        # contents_element = self.rt.findall("Contents")
        # print("got contents element: {}".format(contents_element))

        run_texts = []
        # contents_str = ""
        elements_owned_by_text = self.rt_dict.get_by_owner_guid(self.guid)
        # print("elements owned by {} are:\n{}".format(self.guid, elements_owned_by_text))
        st_texts = [x for x in elements_owned_by_text if x.attrib["class"] == "StText"]
        # print("got StTexts: {}".format(st_texts))
        for st_text in st_texts:
            # print("this StText: {}".format(st_text))
            paragraphs = st_text.findall("Paragraphs")
            # print("paragraphs: {}".format(paragraphs))
            for paragraph in paragraphs:
                objsurs = paragraph.findall("objsur")
                if objsurs is None:
                    # print("Warning: paragraph {} has no objsurs".format(paragraph))
                    continue
                for objsur in objsurs:
                    st_text_para_guid = objsur.attrib["guid"]
                    st_text_paragraph = self.rt_dict[st_text_para_guid]
                    contents = st_text_paragraph.findall("Contents")
                    if len(contents) == 0:
                        continue
                    assert len(contents) == 1, "StTextPara guid {} has more than one contents: {}".format(st_text_para_guid, contents)
                    str_elements = contents[0].findall("Str")
                    assert len(str_elements) == 1, "StTextPara guid {} has more than one Contents>Str: {}".format(st_text_para_guid, str_elements)
                    run_elements = str_elements[0].findall("Run")
                    assert len(run_elements) == 1, "StTextPara guid {} has more than one Contents>Str>Run: {}".format(run_elements)
                    run_text = run_elements[0].text
                    run_texts.append(run_text)
                    # contents_str += run_text
        return run_texts

    def has_contents(self):
        return self.contents is not None

    def __repr__(self):
        return "<Text name='{}' guid={}>".format(self.name, self.guid)
