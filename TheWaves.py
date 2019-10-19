import random


class Observer:
    def __init__(self, genome):
        self.genome = genome

    def evaluate_event(self, event):
        code = event.code
        

class Event:
    def __init__(self, code):
        self.code = code

def get_binary_string(length):
    return "".join([random.choice("01") for _ in range(length)])


if __name__ == "__main__":
    
