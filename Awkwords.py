print("When entering lists of characters, please separate them with spaces, e.g. C p t k m n s kh; e.g. (C) C V L C V N.")

categories = {}
while True:
    next_chars = input("Enter the next category letter, followed by the characters, or press enter to finish.\n").split(" ")
    if next_chars == [""]:
        break
    if next_chars[0] in categories:
        cont = input("This category has already been specified (%s). Would you like to add to it? (y/n) " % repr(categories[next_chars[0]]))
        if cont == "y":
            for i in range(len(next_chars) - 1):
                categories[next_chars[0]].append(next_chars[1 + i])
    else:
        categories[next_chars[0]] = next_chars[1:]
    print("Category %s is %s." % (next_chars[0], repr(categories[next_chars[0]])))

rule = input("Enter the rule.\n").split(" ")

max_words = 1
for q in rule:
    max_words = max_words * len(categories[q])

from random import randrange

k = int(input("How many words would you like to create? "))

words = []
kk = 0
while kk < k:
    word = ""
    for i in rule:
        j = randrange(0,len(categories[i]))
        word = word + categories[i][j]
    if word not in words:
        words.append(word)
        kk += 1
    if len(words) == max_words:
        print("You have produced all possible words.")
        break

print(words)

