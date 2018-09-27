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

def doomsday():
    if do("Doomsday"):
        os.system("python3 Doomsday.py")
    cont()

def rang():
    if do("Rang"):
        os.system("python3 Rang.py 2 y")
    cont()




# routine

doomsday()
rang()

done()
