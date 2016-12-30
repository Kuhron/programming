from random import randrange

while True:
    try:
        digs = int(input("How many digits would you like to try to guess? "))
        if digs % 1 != 0 or digs < 1 or digs > 10:
            print("Input must be between 1 and 10 (inclusive).")
        else:
            break
    except ValueError:
        print("Input must be a positive integer.")

ans = ""
digs_left = [e for e in range(10)]
for i in range(digs):
    d = digs_left[randrange(0,len(digs_left))]
    ans = ans + str(d)
    digs_left.remove(d)
print("Code generated!")

guess_num = 1
while True:
    guess = input("Guess: ")
    if len(guess) != digs:
        print("Guess must consist of %d different digits (you typed %d digits)." % (digs, len(guess)))
    elif guess != ans:
        digs_left = [str(e) for e in range(10)]
        print_counts = True
        for i in guess:
            if i not in digs_left:
                print_counts = False
                print("Guess must consist of %d different digits (you typed some digits more than once)." % digs)
                break
            else:
                digs_left.remove(i)
        wrong_count = 0
        right_count = 0
        for i in range(digs):
            if guess[i] == ans[i]:
                right_count += 1
            else:
                for j in range(digs):
                    if guess[i] == ans[j] and i != j:
                        wrong_count += 1
        if print_counts == True:
            print("%d digits are correct but in the wrong place." % wrong_count)
            print("%d digits are correct and in the correct place." % right_count)
        guess_num += 1
    elif guess == ans:
        break
    
print("Good job! The code was %s! It took you %d guesses for a code of length %d." % (ans, guess_num, digs))
