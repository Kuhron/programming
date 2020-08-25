import random
import string
import time


fp = "RaceConditionOutput.txt"
username = "".join(random.choice(string.ascii_uppercase) for _ in range(random.randint(5,12)))


def waste_time():
    print("User {} going to do processing stuff".format(username))
    for i in range(random.randint(1, 5)):
        print("User {} doing processing stuff... other user should not be able to access file during this time".format(username))
        time.sleep(1)
    print("User {} done with processing stuff".format(username))

def write():
    print("User {} attempting to open file".format(username))
    with open(fp, "a") as f:
        print("User {} successfully opened file".format(username))
        waste_time()
        print("User {} going to write to file now".format(username))
        f.write("User " + username + " says the time is " + str(time.time()) + "\n")
        print("User {} wrote to file, going to do more processing while file is still open".format(username))
        waste_time()
    print("User {} finished writing to file".format(username))

def read():
    print("User {} going to read from file".format(username))
    with open(fp) as f:
        print("User {} has opened file for reading".format(username))
        contents = f.readlines()
        waste_time()
    print("User {} read the file, got {} lines of content".format(username, len(contents)))
    print("User {} finished reading from file".format(username))

def g():
    time.sleep(1)
    f = random.choice([write, read])
    f()
    print("----")

while True:
    g()
