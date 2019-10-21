class SegmentSet(set):
    def __init__(self, element_type, symbol):
        super().__init__()
        self.element_type = element_type
        self.symbol = symbol

    def add(self, item):
        assert type(item) is self.element_type, "can't add item of type {} to SegmentSet of {}".format(type(item), self.element_type)
        super().add(item)
