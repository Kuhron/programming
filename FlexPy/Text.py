class Text:
    def __init__(self, guid, rt):
        self.guid = guid
        self.rt = rt
        self.contents = Text.get_contents_from_rt(rt)

    @staticmethod
    def get_contents_from_rt(rt):
        contents_element = rt.find("Contents")
        print("got contents element: {}".format(contents_element))
        return contents_element

    def has_contents(self):
        return self.contents is not None

