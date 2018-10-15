import os


def cont():
    input("press enter to continue\n")

def do(s):
    a = input(s + " [y]/n: ")
    if a == "n":
        return False
    return True

def done():
    print("done")


# subroutines
# please keep alphabetized

def doomsday():
    if do("Doomsday"):
        os.system("python3 Doomsday.py")
    cont()

def liff():
    if do("Meaning of Liff"):
        os.system("python3 TheMeaningOfLiff.py")
    cont()

def mood():
    if do("Mood Tracker"):
        os.system("python3 MoodTracker.py")
    cont()

def rang():
    if do("Rang"):
        os.system("python3 Rang.py 2 y")
        while do("again?"):
            os.system("python3 Rang.py 2 y")
    cont()


# routine

mood()
doomsday()
rang()
liff()

done()
