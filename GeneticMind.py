# for now just keep memory in an "array" (actually a dict) in RAM, not a file, unless it gets really huge

import random

import numpy as np
import matplotlib.pyplot as plt


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
    CODONS =  ["000", "001", "010", "011", "100", "101", "110", "111"]
    LETTERS = [  "0",   "0",   "R",   "W",   "I",   "S",   "S",   "X"]
    # https://stackoverflow.com/questions/13905741/accessing-class-variables-from-a-list-comprehension-in-the-class-definition
    # https://stackoverflow.com/a/13913933/7376935
    CODONS_TO_LETTERS = (lambda CODONS=CODONS, LETTERS=LETTERS: {k: v for k, v in zip(CODONS, LETTERS)})()
    LETTERS_TO_CODONS = (lambda CODONS=CODONS, LETTERS=LETTERS: {letter: [c for i, c in enumerate(CODONS) if LETTERS[i] == letter] for letter in LETTERS})()

    def __init__(self, binary):
        self.binary = binary
        self.string = Genome.get_string_from_binary(binary)

    def mutate(self):
        r = random.random()
        if r < 0.25:
            return Genome(self.insert())
        elif r < 0.5:
            return Genome(self.delete())
        else:
            return Genome(self.swap())

    def insert(self):
        b = self.binary

        def insert_once(b):
            i = random.randrange(len(b))
            return b[:i] + random.choice("01") + b[i:]

        b = insert_once(b)
        while random.random() < 0.5:
            b = insert_once(b)

        return b

    def delete(self):
        b = self.binary

        def delete_once(b):
            i = random.randrange(len(b))
            return b[:i] + b[i+1:]

        b = delete_once(b)
        while random.random() < 0.5:
            b = delete_once(b)

        return b

    def swap(self):
        b = self.binary

        def swap_once(b):
            i = random.randrange(len(b)-1)
            return b[:i] + b[i+1] + b[i] + b[i+2:]

        b = swap_once(b)
        while random.random() < 0.5:
            b = swap_once(b)

        return b

    @staticmethod
    def from_string(string):
        binary = ""
        for letter in string:
            binary += random.choice(Genome.LETTERS_TO_CODONS[letter])
        return Genome(binary)

    @staticmethod
    def get_string_from_binary(binary):
        l = len(binary)
        n = len(Genome.CODONS[0])
        assert all(len(x) == n for x in Genome.CODONS)
        n_diff = 0 if l % 3 == 0 else 1 if l % 3 == 2 else 2
        binary += "0" * n_diff
        l += n_diff

        result = ""
        n_letters = l // n
        for i in range(n_letters):
            codon = binary[n*i: n*(i+1)]
            result += Genome.CODONS_TO_LETTERS[codon]

        return result

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

    @staticmethod
    def get_function_from_codon(codon):
        return Genome.get_function_from_letter(Genome.CODONS_TO_LETTERS).get(codon, "0")


class Mind:
    def __init__(self, genome):
        self.memory = Memory()
        self.genome = genome

    def process_input(self, input_array):
        queue = []
        queue_repr = ""

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
            queue_repr += letter

        while len(queue) > 1:
            # print(queue_repr)
            last_queue_repr = queue_repr
            for i, item in enumerate(queue):
                if callable(item):
                    n = item.__code__.co_argcount
                    values = Mind.get_next_n_items(queue, n, i)
                    if any(callable(x) for x in values):
                        continue
                    else:
                        result = item(*values)
                        queue = queue[:i] + ([] if result is None else [result]) + (queue[i+n+1:] if i+n+1 < len(queue) else [])
                        queue_repr = queue_repr[:i] + ("" if result is None else "*") + (queue_repr[i+n+1:] if i+n+1 < len(queue_repr) else "")
                        # if None in queue:
                        #     raise RuntimeError("fix this")
                        break
            if queue_repr == last_queue_repr:
                assert all(x == "*" for x in queue_repr), "infinite evaluation in queue of type {}".format(queue_repr)
                queue = [Genome.xor] * (len(queue) - 1) + queue
                queue_repr = "X" * (len(queue) - 1) + queue_repr

        # print(queue[0], self.memory.get_memory_str())

        # need to keep array of functions awaiting values for their parameters, and values that can be used for them
        # also if reach the end of genome but still have functions awaiting values, just append zeros to the value buffer until the expression evaluation terminates

        return queue[0]

    @staticmethod
    def get_next_n_items(queue, n, index_of_func):
        new_queue = queue + [0] * n
        return new_queue[index_of_func + 1: index_of_func + 1 + n]


class RewardSystem:
    # don't instantiate

    @staticmethod
    def evaluate(genome, input_array_list, n_output_lag_terms, plot=True):
        mind = Mind(genome)
        outputs = []
        memory_states = []

        for input_array in input_array_list:
            # add previous output lag terms
            input_array = [
                (0 if len(outputs) < (x+1) else outputs[-(x+1)])
                    for x in range(n_output_lag_terms)
            ] + input_array

            output = mind.process_input(input_array)
            outputs.append(output)
            memory_states.append(mind.memory.memory)

        if plot:
            plt.plot(outputs)
            plt.show()

        reward_function = RewardSystem.mean_stddev_memory

        return reward_function(outputs, memory_states)

    @staticmethod
    def random(outputs, memory_states):
        # just for testing reward system functionality
        return random.random()

    @staticmethod
    def mean_stddev_memory(outputs, memory_states):
        return np.mean([np.std([x for x in memory.values()]) for memory in memory_states])


class Environment:
    def __init__(self):
        # later, can add ability for environment's parameters (generating the distribution of the inputs) to change over time
        self.min = 0
        self.max = 5
        self.array_length = 5
        self.evaluation_lifetime = 1000
        self.n_output_lag_terms = 1

    def get_input_item(self):
        return random.randint(self.min, self.max)

    def get_input_array(self):
        return [self.get_input_item() for _ in range(self.array_length)]

    def evaluate_genome(self, genome, plot=True):
        input_array_list = [self.get_input_array() for _ in range(self.evaluation_lifetime)]
        score = RewardSystem.evaluate(genome, input_array_list, self.n_output_lag_terms, plot=plot)
        return score

    def evolve_genome(self, genome, n_steps):
        candidates = [genome, genome.mutate()]
        n_candidates_to_survive = 5

        for _ in range(n_steps):
            print([x.string for x in candidates])
            scores = [self.evaluate_genome(x, plot=False) for x in candidates]
            top_scores = sorted(zip(scores, [random.random() for _ in range(len(scores))], candidates), reverse=True)  # list of random numbers prevents trying to order Genome objects
            candidates = [candidate for score, _, candidate in top_scores[:n_candidates_to_survive]]
            candidates += [candidate.mutate() for candidate in candidates]

        return candidates[0]


if __name__ == "__main__":
    # test_genome = Genome.from_string("XSS0XSS0SSS0")  # should output 3
    # mind = Mind(test_genome)
    # mind.process_input([])  # make it use real input later, for now just test that genome evaluation is correct

    genome = Genome.from_string("XI0RSIS0WXIS0R0SXI0RS0")  # the one I designed intentionally and implemented in my notebook

    environment = Environment()
    winner = environment.evolve_genome(genome, 20)
    environment.evaluate_genome(winner, plot=True)
