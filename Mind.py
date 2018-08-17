# for now just keep memory in an "array" (actually a dict) in RAM, not a file, unless it gets really huge

class Memory:
    def __init__(self):
        self.memory = {}

    def read(self, index):
        if index not in self.memory:
            # if accessed, initialize
            self.memory[index] = 0

        return self.memory[index]

    def write(self, index, value):
        self.memory[index] = value

    def get_memory_str(self):
        keys = sorted(self.memory.keys())
        return "[ " + ", ".join("{}:{}".format(k, self.memory[k]) for k in keys) + " ]"


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
            "0": Genome.zero,
            "S": Genome.succ,
            "X": Genome.xor,
        }[letter]  # want to raise exception if not found, so we are not having Nones where we should have functions


class Mind:
    def __init__(self, genome):
        self.memory = Memory()
        self.genome = genome

    def process_input(self, input_array):
        queue = []

        for letter in self.genome.string:
            if letter == "W":
                f = lambda i, v: self.memory.write(i, v)
            elif letter == "R":
                f = lambda i: self.memory.read(i)
            elif letter == "I":
                f = lambda i: input_array[i] if i < len(input_array) else 0
            else:
                f = Genome.get_function_from_letter(letter)

            queue.append(f)

        while len(queue) > 1:
            # print(queue)
            for i, item in enumerate(queue):
                if callable(item):
                    n = item.__code__.co_argcount
                    values = Mind.get_next_n_items(queue, n, i)
                    if any(callable(x) for x in values):
                        continue
                    else:
                        result = item(*values)
                        queue = queue[:i] + ([] if result is None else [result]) + (queue[i+n+1:] if i+n+1 < len(queue) else [])
                        # if None in queue:
                        #     raise RuntimeError("fix this")
                        break

        print(queue[0], self.memory.get_memory_str())

        # need to keep array of functions awaiting values for their parameters, and values that can be used for them
        # also if reach the end of genome but still have functions awaiting values, just append zeros to the value buffer until the expression evaluation terminates

    @staticmethod
    def get_next_n_items(queue, n, index_of_func):
        new_queue = queue + [0] * n
        return new_queue[index_of_func + 1: index_of_func + 1 + n]


if __name__ == "__main__":
    test_genome = Genome("XSS0XSS0SSS0")  # should output 3
    mind = Mind(test_genome)
    mind.process_input([])  # make it use real input later, for now just test that genome evaluation is correct

    test_genome = Genome("XI0RSIS0WXIS0R0SXI0RS0")  # the one I designed intentionally and implemented in my notebook
    mind = Mind(test_genome)
    for input_array in [
        [1,0],[1,1],[1,0],[1,1],[3,1],[1,3],[1,1],[3,1],[1,3],[3,1],[1,3],[1,1],[3,1],[1,3],[1,1],[3,1]
    ]:
        mind.process_input(input_array)
