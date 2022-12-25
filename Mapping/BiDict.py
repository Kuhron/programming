# bidirectional dictionary with restriction that the two sides are of different types


class BiDict:
    def __init__(self, type_a, type_b):
        assert type(type_a) is type, f"type_a should be a type, but it is {type_a}, which is of type {type(type_a)}"
        assert type(type_b) is type, f"type_b should be a type, but it is {type_b}, which is of type {type(type_b)}"
        assert type_a is not type_b, f"types must be distinct, but got {type_a} and {type_b}"
        self.type_a = type_a
        self.type_b = type_b
        self.a_to_b = {}
        self.b_to_a = {}

    @staticmethod
    def from_dict(d):
        print(f"constructing BiDict from dictionary with {len(d)} entries")
        if len(d) == 0:
            raise ValueError("cannot initialize BiDict from empty dict because we don't know the key/value types")
        items = list(d.items())
        keys = [kv[0] for kv in items]
        values = [kv[1] for kv in items]
        key_types = set(type(k) for k in keys)
        assert len(key_types) == 1, f"need one key type, but got {key_types}"
        type_a = list(key_types)[0]
        value_types = set(type(v) for v in values)
        assert len(value_types) == 1, f"need one value type, but got {value_types}"
        type_b = list(value_types)[0]
        bd = BiDict(type_a, type_b)
        bd.update(d)
        print(f"-- done constructing BiDict")
        return bd

    def update(self, d):
        for k, v in d.items():
            self.add_pair(k, v)

    def add_pair(self, k, v):
        if type(k) is self.type_a:
            assert type(v) is self.type_b
            a,b = k,v
        elif type(k) is self.type_b:
            assert type(v) is self.type_a
            a,b = v,k
        else:
            raise TypeError(type(k), type(v))
        self.a_to_b[a] = b
        self.b_to_a[b] = a

    def __getitem__(self, index):
        if type(index) is self.type_a:
            return self.a_to_b[index]
        elif type(index) is self.type_b:
            return self.b_to_a[index]
        else:
            raise TypeError(type(index))

    def __setitem__(self, k, v):
        self.add_pair(k, v)

    def __len__(self):
        l = len(self.a_to_b)
        assert len(self.b_to_a) == l
        return l

    def get_sub_dict(self, type1, type2):
        if type1 is self.type_a:
            assert type2 is self.type_b
            return self.a_to_b
        elif type1 is self.type_b:
            assert type2 is self.type_a
            return self.b_to_a
        else:
            raise TypeError(type1, type2)

    def keys(self, type1, type2):
        return self.get_sub_dict(type1, type2).keys()

    def values(self, type1, type2):
        return self.get_sub_dict(type1, type2).values()

    def items(self, type1, type2):
        return self.get_sub_dict(type1, type2).items()

    def __repr__(self):
        return f"<BiDict {self.a_to_b}>"
