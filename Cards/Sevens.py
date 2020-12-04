import random

def r(m, n=None):
    if n is None:
        return random.choice(range(m))
    return random.choice(range(m,n))


# keep in mind that the "-" points for a card are only for being under the card when it is active; "+" points are the deductions for off-class sitters

class OriginalSevensDeck:
    def __init__(self):  # I wish I could just access the class variables in this scope without instantiating
        self.VALUES = ["2","3","4","5","7","J","Q","K","A"]
        self.C0 = ["2","3","4","5"]
        self.C1 = ["A","K","Q","J"]  # note reverse ranking, where J is highest face
        self.SEVEN = ["7"]
        self.SUITS = ["S","H","D","C"]
        self.CARDS = [(self.VALUES[v])+(self.SUITS[-s+3]) for s in range(4) for v in range(-s+3,s+6)]
        self.CLASSES = {"C0":self.C0,"C1":self.C1,"SEVEN":self.SEVEN}
        
        self.POINTS = {
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
        
        # unused experimental point systems
        self.RANDOM_POINTS = {
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
        self.PYRAMID_POINTS = {
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

    def get_class(self, card):
        c = card[0]
        if c in self.C0:
            return "C0"
        if c in self.C1:
            return "C1"
        if c in self.SEVEN:
            return "SEVEN"
        return None

    def sort_by_suit(self, s):
        # 1A 1B 3A 2A 2B 2C --> 
        # 1A 1B 2A 2B 2C 3A --> 
        # A1 B1 A2 B2 C2 A3 --> 
        # A1 A2 A3 B1 B2 C2 --> 
        # 1A 2A 3A 1B 2B 2C

        a = sorted(s, key=lambda x: self.VALUES.index(x[0]))
        c = sorted(a, key=lambda x: self.SUITS.index(x[1]))
        return c


class FullSevensDeck(OriginalSevensDeck):
    def __init__(self):
        self.VALUES = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]
        self.C0 = ["A","2","3","4","5","6"]
        self.C1 = ["8","9","T","J","Q","K"]  # not in reverse order for full deck
        self.SEVEN = ["7"]
        self.SUITS = ["S","H","D","C"]
        self.CARDS = [v + s for v in self.VALUES for s in self.SUITS]
        self.CLASSES = {"C0":self.C0,"C1":self.C1,"SEVEN":self.SEVEN}
        
        self.POINTS = {
            "A":{"+":1,"-":1},
            "2":{"+":2,"-":2},
            "3":{"+":3,"-":3},
            "4":{"+":4,"-":4},
            "5":{"+":5,"-":5},
            "6":{"+":6,"-":6},
            "7":{"+":0,"-":0},
            "8":{"+":1,"-":1},
            "9":{"+":2,"-":2},
            "T":{"+":3,"-":3},
            "J":{"+":4,"-":4},
            "Q":{"+":5,"-":5},
            "K":{"+":6,"-":6},
        }
        

# --- ALGORITHMS FOR SELECTING THE CARD TO PLAY --- #

def random_select(s):
    """
    Simply returns a random member of the set s.
    """
    return random.choice(s)

# basic strategy ideas:
# - if you have the 7 of a suit, play the highest card you have in the class that you are shortest in that suit
# - - (the idea is to force people to play under, while stacking your own cards off class, then 7 as late as possible)
# - if you have low cards, play them as quickly as possible to avoid playing under

select = random_select


# /// ALGORITHMS FOR SELECTING THE CARD TO PLAY /// #

def deal(deck, n_players):
    shuffled = random.sample(deck.CARDS, len(deck.CARDS))
    result = []
    for _ in range(n_players):
        q = int(len(deck.CARDS)/n_players)
        result.append(shuffled[_*q:(_+1)*q])
    return result

def play(n_players, deck, user_plays=True, silent=False):
    scores = [0 for i in range(n_players)]
    user_name = "You"
    CPU_names = ["CPU" + x for x in ["Red","Orange","Yellow","Green","Blue","Purple","Pink","Black","Gray","White","Brown"]]
    player_names = [user_name] + random.sample(CPU_names, n_players-1)

    # track scores by index, print out scores with the name of the player
    last_user_index = 0

    while max(scores) < 100:
        if not silent: print("\nNEW HAND\n")
        hands = deal(deck, n_players)
        user = r(n_players)
        scores = [scores[i%n_players] for i in range((-user+last_user_index),(-user+last_user_index)+n_players)]
        player_names = [player_names[i%n_players] for i in range((-user+last_user_index),(-user+last_user_index)+n_players)]
        if user_plays and not silent:
            input("You will play as index {0} this hand. Press enter when ready.".format(user))
        board = ["S: ","H: ","D: ","C: "]
        points = [{"+":0,"-":0} for i in range(4)] # points outstanding in each suit
        active_classes = ["NONE" for i in range(4)] # active classes in each suit
        active_cards = ["NONE" for i in range(4)] # last on-class or class-setting cards played
        
        card_count = 0
        index = 0
        while card_count < len(deck.CARDS) + 1:
            if not silent:
                print("Board:")
                print("\n".join(board))
                print("Scores:",", ".join([player_names[i]+" : "+str(scores[i]) for i in range(n_players)]))
                if card_count == len(deck.CARDS):
                    # just so the board and scores will be printed at the end of each hand
                    break
                input("press enter to continue")

            if index == user:
                if not silent: print("Your cards:"," ".join(deck.sort_by_suit(hands[user])),"\n")
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

            print("\n{} played {}\n".format(player_names[index], card))

            hands[index].remove(card)
            value_name = card[0]
            suit = deck.SUITS.index(card[1])
            last_card = active_cards[suit]
            p = points[suit]
            cl = deck.get_class(card)
            on_class = cl == active_classes[suit]
            active_class_exists = active_classes[suit] != "NONE"
            is_seven = cl == "SEVEN"
            point_effect = 0 if not on_class else 1 if deck.CLASSES[cl].index(value_name) > deck.CLASSES[cl].index(last_card) else -1

            if not active_class_exists:
                if is_seven:
                    prefix = " "
                else:
                    prefix = "*"
            else:
                if is_seven or point_effect > 0:
                    prefix = "*"
                elif point_effect < 0:
                    prefix = "."
                else:
                    prefix = "~"

            board[suit] += prefix + value_name + " "

            if is_seven:
                scores[index] += p["+"]
                p["+"] = 0
                p["-"] = 0
                active_classes[suit] = "NONE"
                active_cards[suit] = "NONE"
            else:
                if not active_class_exists:
                    p["+"] = deck.POINTS[value_name]["+"]
                    p["-"] = deck.POINTS[value_name]["-"]
                    active_classes[suit] = cl
                    active_cards[suit] = value_name
                elif on_class:
                    if point_effect > 0: # if this card beats the previous card in class
                        scores[index] += p["+"]
                        p["+"] = deck.POINTS[value_name]["+"]
                        p["-"] = deck.POINTS[value_name]["-"]
                        active_cards[suit] = value_name
                    else: # can never be equal (the same card value), so this accounts for penalties for going under
                        scores[index] -= p["-"] # the point values are all stored as absolute value, so subtract here
                        # p["+"] remains unchanged
                        # p["-"] remains unchanged
                else: # if the card is off class
                    # don't change any scores; just add the points
                    p["+"] += deck.POINTS[value_name]["+"]
                    p["-"] += deck.POINTS[value_name]["+"] # this is NOT a typo; the additional penalty for an off-class card is the card's "+" value


            card_count += 1
            index = (index+1) % n_players
            last_user_index = user

    print("\nFinal scores:",", ".join([player_names[i]+" : "+str(scores[i]) for i in range(n_players)]))


def get_int_from_input(description_str, validation_function, invalidation_str):
    while True:
        n = input(description_str + ": ")
        try:
            n = int(n)
        except ValueError:
            print(description_str + " must be a valid int. Please try again.")
            continue  # don't go to the next check yet

        if not validation_function(n):
            print(invalidation_str)
        else:
            return n


if __name__ == "__main__":
    user_is_playing = input("Would you like to play manually? (y to play, n (default) to let computers play themselves.\n").lower() == "y"
    n_cards = get_int_from_input("Number of cards in deck", lambda x: x in [24, 52], "Number of cards must be 24 or 52.")
    n_players = get_int_from_input("Number of players", lambda x: x >= 1 and n_cards % x == 0, "Number of players must be divisor of {}.".format(n_cards))
    
    if n_cards == 24:
        deck = OriginalSevensDeck()
    elif n_cards == 52:
        deck = FullSevensDeck()
    else:
        raise Exception("invalid n_cards was not caught: {}".format(n_cards))

    if user_is_playing:
        play(n_players, deck)
    else:
        # just look at final scores for completely computerized play
        play(n_players, deck, user_plays=False, silent=True)









