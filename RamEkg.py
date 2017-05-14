import time
import numpy as np

from FunctionalFormGui import get_function


class RamFillingList:
    LIST_ITEM = None

    def __init__(self):
        self.lst = []
        self.length = 0

    def to_length(self, new_length):
        if new_length == self.length:
            return
        elif new_length > self.length:
            self.lst += [RamFillingList.LIST_ITEM] * (new_length - self.length)
            self.length = new_length
        elif new_length < self.length:
            self.lst = self.lst[:new_length]
            self.length = new_length

    def __len__(self):
        return self.length


PERIOD_SECONDS = 20  # heart rate about 50 - 80 bpm, so real value can be ~ 1
PERIOD_MILLISECONDS = PERIOD_SECONDS * 1000
MIN_LIST_SIZE = 0
MAX_LIST_SIZE = int(2e8)

f_raw = get_function(0, PERIOD_MILLISECONDS, MIN_LIST_SIZE, MAX_LIST_SIZE)
f = lambda x: f_raw(x % PERIOD_MILLISECONDS)

lst = RamFillingList()
while True:
    t = int(time.time() * 1000)
    list_size = int(f(t))
    lst.to_length(list_size)