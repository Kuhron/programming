import math
import random
import time

import BaseConversion as bc


class FieldLine:
    def __init__(self, n_fields):
        self.fields = [(i, self.get_randint()) for i in range(n_fields)]

    def get_max_field_length(self):
        return max([len(str(x)) for x, _ in self.fields])

    def get_max_value_length(self):
        return max([len(str(x)) for _, x in self.fields])

    def transform(self):
        self.fields = [(field, self.get_randint() if random.random() < 0.001 else value) for field, value in self.fields]

    @staticmethod
    def get_print_int(i):
        return str(i).rjust(3, " ")

    @staticmethod
    def get_randint():
        return random.randint(0, 999)

    def __repr__(self):
        s = ""
        n_fields = len(self.fields)
        for i, (field, value) in enumerate(self.fields):
            f = self.get_print_int(field)
            v = self.get_print_int(value)
            # s += f + ":" + v
            s += v
            if i < n_fields - 1:
                s += " "*2
        return s


class FieldArray:
    def __init__(self, n_lines, n_fields):
        self.lines = [FieldLine(n_fields) for i in range(n_lines)]

    def transform(self):
        for line in self.lines:
            line.transform()

    def __repr__(self):
        return "\n".join([repr(x) for x in self.lines])


class Sequence:
    def __init__(self, base):
        self.base = base
        f = random.choice([
            self.get_arithmetic_sequence,
            self.get_quadratic_sequence,
        ])
        self.seq = f()
        self.index = 0
        self.min_length = 50
        self.max_length = 500

    def get_int(self):
        return random.randint(0, self.base ** 3 - 1)

    def get_arithmetic_sequence(self):
        a, b = sorted([self.get_int() for i in range(2)])
        step = int(math.exp(random.uniform(0, math.log(b-a)))) if b != a else 1
        reverse = random.random() < 0.5
        
        if reverse:
            current = a
            while self.index < self.min_length or current <= b:
                yield current
                current += step
        else:
            current = b
            while self.index < self.min_length or current >= a:
                yield current
                current -= step

    def get_quadratic_sequence(self):
        current = self.get_int()
        for addend in self.get_arithmetic_sequence():
            yield current
            current += addend
            
    def get_print_field(self, x):
        if x is None:
            return " "*3

        x_adj = x % (self.base ** 3)
        x_str_base = bc.from_base_to_base(x_adj, 10, self.base)
        x_str = str(x_str_base).rjust(3, " ")
        return x_str

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index > self.max_length:
            raise StopIteration

        n = next(self.seq)
        if n is None:
            raise StopIteration

        return self.get_print_field(n)
        

class SequenceArray:
    def __init__(self, n_fields, bases):
        self.bases = bases
        self.fields = [None for i in range(n_fields)]
        self.sequences = [None for i in range(n_fields)]

    def transform(self):
        for i in range(len(self.fields)):
            field = self.fields[i]
            seq = self.sequences[i]
            if seq is None:
                if random.random() < 0.01 or all([i is None for i in self.sequences]):
                    self.sequences[i] = self.get_new_sequence()
            else:
                try:
                    self.fields[i] = next(seq)
                except StopIteration:
                    self.fields[i] = None
                    self.sequences[i] = None

    def get_new_sequence(self):
        return Sequence(random.choice(self.bases))

    def __repr__(self):
        delim = " "*2
        return delim.join([x if x is not None else " "*3 for x in self.fields])
    


if __name__ == "__main__":
    MIN_BASE = 2
    MAX_BASE = 62
    
    bases = [10]  # easy
    # bases = [2, 4, 8, 10, 12]  # difficult
    # bases = [i for i in range(MIN_BASE, 36)]  # very difficult
    # bases = [i for i in range(MIN_BASE, MAX_BASE + 1)]  # very very difficult
    delay = 0.3
    
    a = SequenceArray(30, bases)
    while True:
        a.transform()
        print(a)
        time.sleep(delay)
