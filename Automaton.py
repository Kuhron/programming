#cellular automaton

maxwidth = 167

wid = "0"
while int(wid) < 1 or int(wid) > maxwidth:
    wid = input("Width of the automaton (please <= %s)? " % maxwidth)

width = int(wid)

mode = input("Select mode:\n1. normal\nor just press enter for toroidal array\n")
if mode == "1":
    def adjust(column):
        return column
        """if column < 0 or column >= width:
            return None #what to do so that working_row[column] just doesn't exist in this case?"""
elif mode == "":
    def adjust(column):
        return column % width

def text(x):
    if x == 0:
        return " "
    elif x == 1:
        return "X"

import random

seed = [0 for i in range(width)]
seed_ones = input("Which bits should be initially on? Separate with spaces, type \"r\" for random, or press enter for default. " ).split()
if seed_ones == []:
    seed[int((width - 1)/2)] = 1
elif "r" in seed_ones:
    new_seed = []
    for i in range(width):
        change = random.randrange(0, 2)
        new_seed.append(change)
    seed = new_seed
else:
    for i in seed_ones:
        seed[adjust(int(i))] = 1

"""def seed(column):
    if column == (width - 1)/2: #middle of the row
        return 1
    else:
        return 0"""

rule_number = "-1"
while int(rule_number) < 0 or int(rule_number) > 2 ** 8 - 1:
    rule_number = input("Rule number? ")

def prepare_for_printing(row):
    x = ""
    for u in row:
        x = x + text(u)
    return x

delay_raw = input("Delay amount?: ")
try:
    delay = float(delay_raw)
except:
    delay = 0.0

working_row = seed
#for column in range(width):
#    print(text(working_row[column]), end = "")
print(prepare_for_printing(working_row))

"""
def modular_sum(column, spec, reverse = False):
    if reverse == False:
        off = 0
        on = 1
    elif reverse == True:
        off = 1
        on = 0
    speclist = spec.split(",")
    sump = 0
    for s in speclist:
        if ":" in s:
            sump = sump + sum(working_row[min(column+int(s.split(":")[0]),column+int(s.split(":")[1])):max(column+int(s.split(":")[0]),column+int(s.split(":")[1]))+1])
        else:
            sump = sump + working_row[c+int(s)]
    if sump % 2 == 1:
        return on
    elif sump % 2 == 0:
        return off
    #treats edges wrong, last bit copies first one
"""

def elem(column, n): #Wolfram numbering convention for elementary cellular automata
    nbin = "{0:b}".format(n)
    nbin = ("0" * (8 - len(str(nbin)))) + str(nbin)
    cases = {}
    for i in range(8):
        x = str("{0:b}".format(7-i))
        x = ("0" * (3 - len(x))) + x
        cases[x] = nbin[i]
    determiner = "".join([str(working_row[adjust(column+i)]) for i in range(-1, 2)])
    return int(cases[determiner])

def rule(column, working_row):
    return elem(adjust(column), int(rule_number))
    #if (working_row[adjust(column+1)] == 1) != (working_row[adjust(column-1)] == 1):

import time

while True:
    for column in range(width):
        k = rule(adjust(column), working_row)
        working_row.append(k)
    working_row = working_row[width:]
    print(prepare_for_printing(working_row))
    time.sleep(delay)

