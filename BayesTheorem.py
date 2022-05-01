import random


class Card:
    COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "tan", "black", "white"]
    COLOR_ABBREVIATIONS = list("ROYGBIVTKW")
    NUMBERS = list(range(10))
    COLOR_TO_LETTER = dict(zip(COLORS, COLOR_ABBREVIATIONS))
    LETTER_TO_COLOR = dict(zip(COLOR_ABBREVIATIONS, COLORS))
    N_POSSIBLE_CARDS = len(COLORS) * len(NUMBERS)
    COLOR_TO_INDEX = dict((c, i) for i, c in enumerate(COLORS))
    NUMBER_TO_INDEX = dict((n, i) for i, n in enumerate(NUMBERS))

    def __init__(self, color, number):
        assert color in Card.COLORS
        self.color = color
        self.letter = Card.COLOR_TO_LETTER[color]
        assert number in Card.NUMBERS
        self.number = number

    def __eq__(self, other):
        return self.color == other.color and self.number == other.number

    def val(self):
        color_index = Card.COLOR_TO_INDEX[self.color]
        number_index = Card.NUMBER_TO_INDEX[self.number]
        n_numbers = len(Card.NUMBERS)
        return (color_index * n_numbers) + number_index

    def __gt__(self, other):
        return self.val() > other.val()

    @staticmethod
    def random():
        color = random.choice(Card.COLORS)
        number = random.choice(Card.NUMBERS)
        return Card(color, number)

    @staticmethod
    def get_random_set(n):
        if n < 0 or n > Card.N_POSSIBLE_CARDS:
            raise ValueError(f"n must be between 0 and max number of cards ({Card.N_POSSIBLE_CARDS})")
        if n > Card.N_POSSIBLE_CARDS / 2:  # don't put equals here or passing exactly max/2 will create infinite recursion
            # optimize by instead choosing cards to NOT include, and including all others
            n_cards_to_exclude = Card.N_POSSIBLE_CARDS - n
            cards_to_exclude = Card.get_random_set(n_cards_to_exclude)
            return [c for c in Card.get_all_cards() if c not in cards_to_exclude]
        else:
            res = set()
            while len(res) < n:
                res.add(Card.random())
            return res

    @staticmethod
    def get_random_with_repetition(n):
        return [Card.random() for i in range(n)]

    @staticmethod
    def get_all_cards():
        return [Card(color, number) for color in Card.COLORS for number in Card.NUMBERS]

    def __repr__(self):
        return f"{self.letter}{self.number}"

    def __hash__(self):
        return hash(repr(self))

    @staticmethod
    def print_card_count_table(cards):
        # takes a list of cards and prints a table showing how much of each color/number (and of each combo thereof, i.e., each card) there are
        color_to_count = {color: 0 for color in Card.COLORS}
        number_to_count = {number: 0 for number in Card.NUMBERS}
        card_to_count = {card: 0 for card in Card.get_all_cards()}
        for card in cards:
            color_to_count[card.color] += 1
            number_to_count[card.number] += 1
            card_to_count[card] += 1

        # now get the table, columns are numbers and rows are colors
        # so header row shows each number as a label first
        # there will also be a second header row (and similarly a second margin column) showing the count of that number across all colors
        n_colors = len(Card.COLORS)
        n_numbers = len(Card.NUMBERS)
        n_rows = n_colors + 3
        n_cols = n_numbers + 3
        table = [["" for col in range(n_cols)] for row in range(n_rows)]

        # create header row
        for number_i in range(n_numbers):
            ci = number_i + 3  # margin's columns are empty in the header
            table[0][ci] = str(Card.NUMBERS[number_i])
        # create margin column
        for color_i in range(n_colors):
            ri = color_i + 3  # header's rows are empty in the margin column
            table[ri][0] = Card.COLOR_ABBREVIATIONS[color_i]
        # create number count row
        for number_i in range(n_numbers):
            ci = number_i + 3
            table[1][ci] = str(number_to_count[Card.NUMBERS[number_i]])
        assert sum(number_to_count.values()) == len(cards)
        # create color count column
        for color_i in range(n_colors):
            ri = color_i + 3
            table[ri][1] = str(color_to_count[Card.COLORS[color_i]])
        assert sum(color_to_count.values()) == len(cards)
        # fill in individual card counts
        for number_i in range(n_numbers):
            for color_i in range(n_colors):
                ci = number_i + 3
                ri = color_i + 3
                color = Card.COLORS[color_i]
                number = Card.NUMBERS[number_i]
                count = card_to_count.get(Card(color, number), 0)
                table[ri][ci] = str(count) if count > 0 else ""

        # decide how wide to print the columns
        max_col_width = max(len(item) for row in table for item in row)

        # print the table
        print("\n---- Card table ----")
        for row in table:
            s = ""
            for item in row:
                s += item.rjust(max_col_width) + " "
            print(s)
        print("--------------------")



if __name__ == "__main__":
    n = random.randint(1, 100)
    # cards = Card.get_random_set(n)
    cards = Card.get_random_with_repetition(n)
    print(f"got {n} cards:")
    print(sorted(cards))
    Card.print_card_count_table(cards)

    # suppose for now we are estimating the probabilities of everything (every color, every number)
    # start with knowing nothing, including any probabilities of stuff given other stuff
    # try to avoid keeping an actual record of observations; if it's pure Bayes, then just keep priors for each round
    # including those for the given ones?? should be possible in principle, but how to operationalize it?

    # have a training period in which we learn probabilities of stuff by looking at cards one at a time
    # can't remember what we've seen before, but we can keep the priors stored
    # then, post-training, there's a secret card that we will have to estimate about
    # we are given one attribute of the card and have to guess the other (how likely each option for it is)

    priors = {}
    n_observations = 100
    for obs_i in range(n_observations):
        # draw a card from the deck, with replacement
        card = random.choice(cards)
        color = card.color
        number = card.number
        print(f"drew card {card}")

        # update all priors about colors, numbers, and each given the other, based on seeing this card
        # should also incorporate this color observation into all priors about other colors, how?

        old_p_color = priors.get(color)
        old_p_number = priors.get(number)
        old_p_color_given_number = priors.get((color, number))
        old_p_number_given_color = priors.get((number, color))
        print("priors:", priors)
        print(f"p(c) = {old_p_color}\np(n) = {old_p_number}\np(c|n) = {old_p_color_given_number}\np(n|c) = {old_p_number_given_color}\n")

        # for a previously unobserved thing, what to do? e.g. a never-before-seen color
        # I would want to weight it based on how many cards of other colors I've seen before, but I'm not allowing us to do that here
        # so what prior to use? where do the original priors come from? feels like infinite regress

        # good idea to have an estimator for "other" along any dimension, e.g. be constantly updating the probability of seeing a never-before-seen color
        # hopefully this can be a proxy for how many observations you've had
        # e.g. if you have prior of red being twice as likely as blue, and a 10% chance of never-before-seen color coming up, you can assign higher prior to yellow after seeing only one yellow, than you would if the never-before-seen color probability had been only 0.01%
        # so every time you see a known color, the never-before-seen color probability needs to shrink


        input("press enter for next")
