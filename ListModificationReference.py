import sys


def insert_numbers(lst, before_number, after_number, new_number):
    result = []
    for i in range(len(lst)):
        if i >= len(lst) - 1:
            result += [lst[i]]
        elif lst[i] == before_number and lst[i + 1] == after_number:
            result += [lst[i], new_number]
        else:
            result += [lst[i]]
    return result

def delete_numbers(lst, before_number, after_number):
    return [x for x in replace_numbers(lst, before_number, after_number, None) if x is not None]

def replace_numbers(lst, before_number, after_number, new_number):
    result = [lst[0]]
    for i in range(len(lst) - 1):
        if i >= len(lst) - 2:
            result += [lst[i + 1]]
        elif lst[i] == before_number and lst[i + 2] == after_number:
            result += [new_number]
        else:
            result += [lst[i + 1]]
    return result



lst = [int(i) for i in sys.argv[1:]]

print("original   :", lst)

print("insertion  :", insert_numbers(lst, 0, 1, 999))
print("deletion   :", delete_numbers(lst, 0, 1))
print("replacement:", replace_numbers(lst, 0, 1, 999))