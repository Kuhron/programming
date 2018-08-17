# for now just keep memory in an "array" (actually a dict) in RAM, not a file, unless it gets really huge

class Memory:
    def __init__(self):
        self.memory = {}

    def read(self, index):
        return self.memory.get(index, 0)

    def write(self, index, value):
        self.memory[index] = value


class Genome:
    def __init__(self, string):
        self.string = string

    @staticmethod
    def zero():
        return 0

    @staticmethod
    def succ(x):
        return x + 1

    # do not use; mod by zero could occur
    # @staticmethod
    # def addmod(a, b, m):
    #     return (a + b) % m

    @staticmethod
    def xor(a, b):
        return a ^ b

    @staticmethod
    def get_function_from_letter(letter):
        return {
            "Z": Genome.zero,
            "S": Genome.succ,
            "X": Genome.xor,
        }.get(letter)


class Mind:
    def __init__(self, genome):
        self.memory = Memory()
        self.genome = genome

    def process_input(self, input_array):
        queue = []

        for letter in self.genome:
            if letter == "W":
                f = self.memory.write
            if letter == "R":
                f = self.memory.read
            else:
                f = Genome.get_function_from_letter(letter)

            queue.append(f)

        while len(queue) > 1:

            n_args = f.__code__.co_argcount

        # need to keep array of functions awaiting values for their parameters, and values that can be used for them
        # also if reach the end of genome but still have functions awaiting values, just append zeros to the value buffer until the expression evaluation terminates
        raise

    @staticmethod
    def get_next_n_items(queue, n, index_of_func):
        new_queue = queue + [0] * n
        return new_queue[index_of_func + 1: index_of_func + 1 + n]


if __name__ == "__main__":
    test_genome = "XSSZSXSSZSSSZ"  # should output 3


