# NOTE: only works in command line, NOT IN IDLE

# since command line seems to be running Python 2.x (why?)
from __future__ import print_function

import msvcrt
import random
import sys

# DVORAK KEYBOARD TYPES
# ` 1 2 3 4 5 6 7 8 9 0 [ ]     # 0 1 2 3 0 1 2 3 0 1 0 3 1
#    ' , . p y f g c r l / = \  #    3 0 2 0 1 2 3 3 2 0 3 1 0
#     a o e u i d h t n s -     #     1 3 1 1 1 0 0 0 2 3 1
#      ; q j k x b m w v z      #      3 1 2 3 0 2 1 3 2 2
#
#   / * -  #   3 2 1
# 7 8 9 +  # 3 0 1 3
# 4 5 6 +  # 0 1 2 3
# 1 2 3    # 1 2 3
# 0 0 .    # 0 0 2

d = {0:"0", 1:"1", 2:"2", 3:"3"}
e = {"":"",
     "0":"", "1":"", "2":"", "3":"",
     "01":"", "02":"", "03":"",
     "10":"", "12":"", "13":"",
     "20":"", "21":"", "23":"",
     "30":"", "31":"", "32":"",
     "00":"0", "11":"1", "22":"2", "33":"3", # this row lets you ask the "type"
     "010":"E", "011":"T", "012":"A", "013":"O",
     "020":"4", "021":"5", "022":"6", "023":"7",
     "030":"8", "031":"9", "032":"+", "033":"-",
     "100":"I", "101":"N", "102":"S", "103":"H",
     "120":"R", "121":"D", "122":"L", "123":"U",
     "130":"C", "131":"M", "132":"F", "133":"W",
     "200":"Y", "201":"P", "202":"V", "203":"B",
     "210":"G", "211":"K", "212":"Q", "213":"J",
     "230":"X", "231":"Z", "232":"/", "233":"*",
     "300":"\n", "301":" ", "302":"!", "303":"@",
     "310":"#", "311":"$", "312":"%", "313":"^",
     "320":"&", "321":"(", "322":")", "323":"="
}

while True:
    s = ""
    while len(s) < 3:
        #q = ord(msvcrt.getch())
        q = random.randrange(4,8)
        if q == 3: # Ctrl-C
            sys.exit()
        if q == 13: # Enter, for when you're lost
            break
        r = q % 4
        if r in d:
            s += d[r]
            if len(s) >= 2:
                if s[-1] == s[-2]:
                    break
    if s in e:
        print(e[s], end = "")
    else:
        print(" ?("+s+") ", end = "")
    

"""
print("Choose a number from 1 to 7")
u = ""
while u not in ["1", "2", "3", "4", "5", "6", "7"]:
    u = ord(msvcrt.getch())
    if u
    print(u)
    if u == "1":
        print("one")
"""
