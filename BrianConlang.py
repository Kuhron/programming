s = input("Plaintext: ")
t = ""
for i in range(len(s)):
    if s[i] in "aeiou":
        t = t + "u"
    elif s[i] in "AEIOU":
        t = t + "U"
    elif s[i] == "y":
        if i == 0:
            t = t + s[i]
        elif s[i-1] == " ":
            t = t + s[i]
        elif s[i-1] in "aeiouAEIOU" and s[i+1] == " ":
            t = t + "u"
        elif s[i-1] in "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ":
            t = t + "u"
    else:
        t = t + s[i]
print(t)
