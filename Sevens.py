import random

def r(m,n=None):
    if n == None:
        return random.choice(range(m))
    return random.choice(range(m,n))

VALUES = ["2","3","4","5","7","J","Q","K","A"]
NUMBERS = ["2","3","4","5"]
FACES = ["A","K","Q","J"]
SEVEN = ["7"]
SUITS = ["S","H","D","C"]
CARDS = [(VALUES[v])+(SUITS[-s+3]) for s in range(4) for v in range(-s+3,s+6)]
CLASSES = {"NUMBERS":NUMBERS,"FACES":FACES,"SEVEN":SEVEN}

DEFAULT_POINTS = {
    "2":{"+":2,"-":1},
    "3":{"+":3,"-":1},
    "4":{"+":4,"-":1},
    "5":{"+":5,"-":1},
    "7":{"+":0,"-":0},
    "J":{"+":7,"-":1},
    "Q":{"+":7,"-":1},
    "K":{"+":7,"-":1},
    "A":{"+":1,"-":1}
}
RANDOM_POINTS = {
    "2":{"+":r(8),"-":r(8)},
    "3":{"+":r(8),"-":r(8)},
    "4":{"+":r(8),"-":r(8)},
    "5":{"+":r(8),"-":r(8)},
    "7":{"+":0,"-":0},
    "J":{"+":r(8),"-":r(8)},
    "Q":{"+":r(8),"-":r(8)},
    "K":{"+":r(8),"-":r(8)},
    "A":{"+":r(8),"-":r(8)}
}
PYRAMID_POINTS = {
    "2":{"+":1,"-":1},
    "3":{"+":2,"-":2},
    "4":{"+":4,"-":4},
    "5":{"+":8,"-":8},
    "7":{"+":0,"-":0},
    "J":{"+":8,"-":8},
    "Q":{"+":4,"-":4},
    "K":{"+":2,"-":2},
    "A":{"+":1,"-":1}
}
# keep in mind that the "-" points for a card are only for being under the card when it is active; "+" points are the deductions for off-class sitters
POINTS = DEFAULT_POINTS
# print(POINTS)

# print(sorted(CARDS))
# for s in range(4):
#     print([i for i in filter(lambda x: SUITS[s] in x, CARDS)])

# --- ALGORITHMS FOR SELECTING THE CARD TO PLAY --- #

def random_select(s):
    """
    Simply returns a random member of the set s.
    """
    return random.choice(s)

select = random_select

# /// ALGORITHMS FOR SELECTING THE CARD TO PLAY /// #

def deal(n_players):
    """
    Takes the number of players (must be a divisor of 24).
    Returns a list of lists which are the cards held by each player.
    """
    n = n_players
    if 24.0/n % 1 != 0 or n <= 0 or abs(n) == float("inf") or n != n: # no funky float stuff happens for the correct numbers, so we're good
        raise ValueError("Number of players must be a divisor of 24.")
    if n == 24:
        raise ValueError("That's too many people. While this is possible, it is deterministic and won't be much fun.")

    shuffled = random.sample(CARDS,24)
    result = []
    for _ in range(n_players):
        q = int(24.0/n_players)
        result.append(shuffled[_*q:(_+1)*q])
    return result

# print(deal(4))

def get_class(card):
    c = card[0]
    if c in NUMBERS:
        return "NUMBERS"
    if c in FACES:
        return "FACES"
    if c in SEVEN:
        return "SEVEN"
    return None

def sort_by_suit(s):
    # 1A 1B 3A 2A 2B 2C --> 
    # 1A 1B 2A 2B 2C 3A --> 
    # A1 B1 A2 B2 C2 A3 --> 
    # A1 A2 A3 B1 B2 C2 --> 
    # 1A 2A 3A 1B 2B 2C

    a = sorted(s,key=lambda x:VALUES.index(x[0]),reverse=True)
    # b = [i[::-1] for i in a]
    c = sorted(a,key=lambda x:SUITS.index(x[1]))
    return c #[i[::-1] for i in c]

def play(n_players,user_plays=True,silent=False):
    scores = [0 for i in range(n_players)]
    user_name = "You"
    CPU_names = ["CPU"+i for i in ["Red","Orange","Yellow","Green","Blue","Purple","Pink","Black","Grey","White","Brown"]]
    player_names = [user_name] + random.sample(CPU_names, n_players-1)
    # print("GO FIX THE BUG ABOUT SCORE INDEX, THE PLAYERS KEEP GETTING REARRANGED") # fixed

    # track scores by index, print out scores with the name of the player
    last_user_index = 0

    while max(scores) < 100:
        if not silent: print("\nNEW HAND\n")
        hands = deal(n_players)
        user = r(n_players)
        scores = [scores[i%n_players] for i in range((-user+last_user_index),(-user+last_user_index)+n_players)]
        player_names = [player_names[i%n_players] for i in range((-user+last_user_index),(-user+last_user_index)+n_players)]
        if user_plays and not silent:
            waste = input("You will play as index {0} this hand. Press enter when ready.".format(user))
        # user_cards = hands[user]
        board = ["S: ","H: ","D: ","C: "]
        points = [{"+":0,"-":0} for i in range(4)] # points outstanding in each suit
        active_classes = ["NONE" for i in range(4)] # active classes in each suit
        active_cards = ["NONE" for i in range(4)] # last on-class or class-setting cards played
        
        card_count = 0
        index = 0 #index = (-1*user) % n_players # <- this last thing was redundant with the rotating of the scores list
        while card_count < 24:
            if not silent:
                print("----")
                print("Board:")
                print("\n".join(board))
                print("Scores:",", ".join([player_names[i]+" : "+str(scores[i]) for i in range(n_players)]))

            if index == user:
                if not silent: print("Your cards:"," ".join(sort_by_suit(hands[user])),"\n")
                # user plays
                while True:
                    if user_plays and not silent:
                        card = input("Select a card: ").upper()
                    else:
                        card = random_select(hands[user])
                    if card not in hands[user] and not silent:
                        print("That is not a card you have.")
                    else:
                        break
            else:
                # computers play
                card = select(hands[index])

            hands[index].remove(card)
            value_name = card[0]
            suit = SUITS.index(card[1])
            # if len(board[suit]) == 3: # what they start as
            #     last_card = None
            # else:
            #     last_card = board[suit][-2] # value only
            last_card = active_cards[suit]
            board[suit] += value_name + " "
            p = points[suit]
            cl = get_class(card)

            # if active_classes[suit] == "NONE":
            #     active_classes[suit] = cl
            #     # no points can be added or subtracted in this situation

            if value_name == "7":
                scores[index] += p["+"]
                p["+"] = 0
                p["-"] = 0
                active_classes[suit] = "NONE"
                active_cards[suit] = "NONE"
            else: # non-seven
                if active_classes[suit] == "NONE": # no class
                    p["+"] = POINTS[value_name]["+"]
                    p["-"] = POINTS[value_name]["-"]
                    active_classes[suit] = cl
                    active_cards[suit] = value_name
                elif cl == active_classes[suit]: # if the card is on class
                    if CLASSES[cl].index(value_name) > CLASSES[cl].index(last_card): # if this card beats the previous card in class
                        scores[index] += p["+"]
                        p["+"] = POINTS[value_name]["+"]
                        p["-"] = POINTS[value_name]["-"]
                        active_cards[suit] = value_name
                    else: # can never be equal (the same card value), so this accounts for penalties for going under
                        scores[index] -= p["-"] # the point values are all stored as absolute value, so subtract here
                        # p["+"] remains unchanged
                        # p["-"] remains unchanged
                else: # if the card is off class
                    # don't change any scores; just add the points
                    p["+"] += POINTS[value_name]["+"]
                    p["-"] += POINTS[value_name]["+"] # this is NOT a typo; the additional penalty for an off-class card is the card's "+" value


            card_count += 1
            index = (index+1) % n_players
            last_user_index = user

    print("\nFinal scores:",", ".join([player_names[i]+" : "+str(scores[i]) for i in range(n_players)]))

# play(int(input("How many players? ")),user_plays=False,silent=True) # just look at final scores for completely computerized play
play(int(input("How many players? ")))









