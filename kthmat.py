# http://kthmat.blogspot.com/2015/07/ok-qxf-fdfqayugcfgb-igiijwh-idaeo-bome-y.html


import string


def caesar(plaintext, shift):
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted_alphabet)
    return plaintext.translate(table)




with open("words.txt") as f:
    words = [x.strip() for x in f.readlines()]

with open("kthmat.txt") as f:
    ciphertext = f.read()


for i in range(26):
    newtext = caesar(ciphertext, -i)
    if True: # any([x.strip() in words for x in newtext.split()]):
        print(i)
        print(newtext)
        print("\n\n")
