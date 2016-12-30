import random

card_values = [str(i) for i in range(2,10)] + ["T","J","Q","K","A"]
card_suits = ["C","D","H","S"]

cards = [i+j for j in card_suits for i in card_values]
#print(cards)

hands = ["high card","pair","two pair","three of a kind","straight","flush","full house","four of a kind","straight flush"]
stages = ["draw","flop","turn","river"]

def ask_hope():
    a = input("What kind of hand are you hoping for?")
    if a not in hands:
        print("Invalid hand.")
        return None
    return a

def select_cards():
    silver = random.sample(cards,7)
    return {"pocket":silver[:2], "draw":[], "flop":silver[2:5], "turn":silver[2:6], "river":silver[2:]}

def show_cards(lst):
    print("  ".join([c[0] for c in lst]))
    print("  ".join([c[1] for c in lst]))

d = select_cards()
pocket = d["pocket"]
show_cards(pocket)

def display_table(dct, stage):
    print("Displaying Table")
    print("\nYour pocket:")
    show_cards(dct["pocket"])
    print("\nTable:")
    show_cards(dct[stage])

# for s in stages:
#     display_table(d,s)
#     ktjqxhbkhqjceueaou = input()

def is_straight(lst):
    jungle = sorted([card_values.index(c[0]) for c in lst])
    print(jungle)
    return [a - jungle[0] for a in jungle] == [u for u in range(5)] or jungle == [u for u in range(4)] + [12]

def value_counts(lst):
    v = [c[0] for c in lst]
    result = {}
    for val in card_values:
        result[val] = v.values().count(val)
    return result

def suit_counts(lst):
    s = [c[1] for c in lst]
    result = {}
    for suit in card_suits:
        result[suit] = s.count(suit)
    return result

def is_flush(lst):
    return 5 in suit_counts(lst).values()

def is_straight_flush(lst):
    return is_straight(lst) and is_flush(lst)

def is_four_of_a_kind(lst):
    return 4 in value_counts(lst).values()

def is_three_of_a_kind(lst):
    return 3 in value_counts(lst).values()

def is_pair(lst):
    return 2 in value_counts(lst).values()

def is_full_house(lst):
    return is_pair(lst) and is_three_of_a_kind(lst)

def is_two_pair(lst):
    return value_counts(lst).count(2) == 2

def is_high_card(lst):
    return True

def verify(lst, hand_type):
    if hand_type not in hands:
        print("Invalid hand.")
        return None

    # from http://stackoverflow.com/questions/7936572/python-call-a-function-from-string-name
    f_name = "is_"+hand_type.replace(" ","_")
    possibles = globals().copy()
    possibles.update(locals())
    f = possibles.get(f_name)
    if not f:
         raise Exception("Method %s not implemented" % f_name)
    # end
    
    return f(lst)

h = ["6S","7S","3S","4S","5S"]
print(h)
for catamaran in hands:
    print(catamaran,":",verify(h, catamaran))