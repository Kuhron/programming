import pandas as pd
import random

df = pd.read_csv("Constellations.csv")

# print(df)
# print(df.index)
# print(df.columns)
# print(df.ix[4,"Constellation"])

def practice_one_set(n):
    print("set", n, "(#{}-{})".format(4*n, 4*n+4))
    for x in range(4*n, 4*n+4):
        row = df.ix[x,:]
        inp = input("What is constellation #{}? ".format(row["Rank"]))
        print("Correct!" if inp == row["Constellation"].strip() else "Incorrect")
        print(row["Rank"], row["Constellation"], "\n")

def practice_random_set():
    n = random.randrange(22)
    practice_one_set(n)

def practice_all_sets():
    for n in range(22):
        practice_one_set(n)

if __name__ == "__main__":
    practice_all_sets()
