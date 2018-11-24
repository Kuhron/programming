# test reddit bot

import ast
import praw
import random
import threading
import time


d = ast.literal_eval(open("_secret_TestBotInfo.txt").read().strip())
client_id = d["client_id"]
client_secret = d["client_secret"]
username = d["username"]
password = d["password"]
r = praw.Reddit(client_id=client_id, client_secret=client_secret, password=password, user_agent="test by /u/Wxyo", username=username)
# r.login(username, d["password"])
# verify logged in
print("logged in as "+ r.user.me())

def test_message():
    # r.send_message(
    #     d["user"], 
    #     "Test Message at {0}".format(time.strftime("%y-%m-%d-%H:%M:%S")),
    #     "Here's a random number from 1 to 100: {0}.\n- your test bot".format(random.randrange(1,101))
    # )
    r.redditor("Wxyo").message("TEST", "This happened!")

def analyze_memeconomy():


if __name__ == "__main__":
    # test_message()
