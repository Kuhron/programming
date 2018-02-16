import time

from Cards.Blackjack.CountingAndBettingSystem import CountingAndBettingSystem
from Cards.Blackjack.Shoe import Shoe


def generate_shoes():
    n_decks = 1
    ratio_dealt = 1
    while True:
        shoe = Shoe(n_decks, ratio_dealt)
        shoe.shuffle()
        cards = shoe.deal()
        for card in cards:
            if shoe.get_n_cards_left() > 0:
                yield shoe
        
        n_decks = 1 if n_decks == 8 else n_decks + 1  # screw modulo
        
def write_shoe_to_file(shoe):
    tc = shoe.get_hi_lo_count()
    tc = round(tc, 1)  # most precision you will ever need, probably (true count to nearest 0.1)
    if abs(tc) < 1e-3:  # i hate floats, just put int 0 in the filename
        tc = 0
    fp = "Bootstraps/tc_{tc}_{now}.shoe".format(tc=tc, now=int(time.time() * 1e6))
    with open(fp, "w") as f:
        f.write("".join(x.value for x in shoe.cards_left))


def run():
    shoe_generator = generate_shoes()
    for shoe in shoe_generator:
        write_shoe_to_file(shoe)


if __name__ == "__main__":
    # run()
    print("doing nothing. uncomment run function if you want to create hundreds of megs of shoe files")
