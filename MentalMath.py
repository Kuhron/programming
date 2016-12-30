import random

from numpy import isclose

def dig():
    """
    Gets a random digit as a string.
    """

    return str(random.choice(range(10)))

def get_arg(digs_before, digs_after):
    """
    Gets a number with max digs_before digits before decimal and max digs_after digits after decimal.
    Please note: this can return 0 in rare cases!
    """

    if digs_before > 0:
        b = random.choice(range(1,digs_before+1))
        before_stuff = [dig() for i in range(b)]
    else:
        before_stuff = ["0"]
    if digs_after > 0:
        a = random.choice(range(1,digs_after+1))
        after_stuff = [dig() for i in range(a)]
        decimal = ["."]
    else:
        after_stuff = []
        decimal = []

    result = "".join(before_stuff + decimal + after_stuff)
    # print("first result:",result)

    # strip leading and trailing zeros
    while len(result) > 1 and result[0] == "0" and result[1] != ".":
        result = result[1:]
    if "." in result:
        while result[-1] == "0":
            result = result[:-1]
        if result[-1] == ".": # everything after the decimal was zeros
            result = result[:-1]

    # print("final result:",result)
    return result

# for i in range(15):
#     print(get_arg(1,0))

def give_problem(digs_before, digs_after, operation):
    arg2 = get_arg(digs_before,digs_after)
    while operation == "/" and isclose(float(arg2), 0.0):
        arg2 = get_arg(digs_before,digs_after)
    
    problem = get_arg(digs_before,digs_after) + " " + operation + " " + arg2 + (".0" if operation == "/" and "." not in arg2 else "")

    given_answer = ""
    while given_answer == "":
        given_answer = input(problem + " = ")
    given_answer = float(given_answer)
    right_answer = eval(problem)
    if right_answer != 0:
        percent_error = 100*float(given_answer-right_answer)/right_answer
    else:
        percent_error = 0 if given_answer == 0 else float("inf")*abs(given_answer)/float(given_answer)

    if operation != "/":
        if given_answer == right_answer:
            success = True
        else: # elif type(right_answer) == int: <- wait, what was I thinking here?
            success = isclose(given_answer, right_answer) # allows for stupid float approximate equality
        # else:
        #     success = False
    else:
        success = abs(percent_error) <= 1

    if success:
        print("Correct! Answer = {0}".format(right_answer))
    else:
        print("Wrong! ({1:.2f}% off) Answer = {0}".format(right_answer,percent_error))

    print()
    return success

def main():
    while True:
        while True:
            try:
                max_digits_before_decimal = int(input("Max digits before decimal? "))
                break
            except ValueError:
                print("invalid input")

        while True:
            try:
                max_digits_after_decimal = int(input("Max digits after decimal? "))
                break
            except ValueError:
                print("invalid input")

        if max_digits_before_decimal <= 0 and max_digits_after_decimal <= 0:
            print("You gotta have some digits, yo.")
            continue
        else:
            break

    operations = []
    while operations == []:
        print("Please select the operations to use by responding with anything other than a newline. ")
        if input("Addition? ") != "":
            operations.append("+")
        if input("Subtraction? ") != "":
            operations.append("-")
        if input("Multiplication? ") != "":
            operations.append("*")
        if input("Division? ") != "":
            operations.append("/")
        if input("Exponentiation? (breaks code a lot) ") != "":
            operations.append("**")

        if operations == []:
            print("You gotta have some operations, yo.")
            continue
        else:
            break

    print("\nTime for problems! Good luck." + (" For division problems you must be within 1%." if "/" in operations else ""))
    problems_asked = 0
    problems_right = 0
    
    while True:
        try:
            if give_problem(max_digits_before_decimal, max_digits_after_decimal, random.choice(operations)):
                problems_right += 1
            if True: # readability maybe? at least makes accidental un-indenting harder
                problems_asked += 1
        except KeyboardInterrupt:
            print("\nYou have quit the game. Final score: {0}/{1} problems right ({2:.0f}%)".format(
                problems_right,problems_asked,float(problems_right)/problems_asked*100))
            break

main()