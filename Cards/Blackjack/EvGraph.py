# import stuff


def get_outcomes_in_isolation(player_hand, dealer_card, deck):
    # not for use in the simulation itself (don't break it)
    # but for finding EV of each action in different situations
    outcomes = {"H": [], "S": [], "D": [], "P": []}
    tc = deck.get_true_count()
    deck = [c for c in deck]
    for _ in range(10):
        random.shuffle(deck)
        # relevant_cards = []  # get the next cards up to a hard total of at least 42, since this would be enough to guarantee that both player and dealer bust, and can't re-draw the same ones from the generator
        # relevant_card_hard_total = 0
        # while relevant_card_hard_total < 42:
        #     c = next(deck)
        #     relevant_card_hard_total += c.get_hard_value()
        #     relevant_cards.append(c)

        dealer_downcard = deck[0]

        actions = "HSDP" if player.has_pair() else "HSD"
        for action in actions:
            # ignore surrender since it is always -0.5
            # how to get EV of a single action? if action leads to busting, record a 0, else record EV of resulting hand and deck
            # shuffle deck to get distribution of outcomes
            # so this is a recursive EV calculation. What are the base cases? (player 21 vs dealer bust/push/less, player bust vs dealer anything, stuff like that) Is there a better way?
            player_hand.take_action(action)
            if player_hand.is_busted():
                outcome = -1  # lose regardless of what happens to dealer
            elif is_push:
                outcome = 0
            elif stand and dealer_busts:
                outcome = +1
            else:
                # player did not bust, so either hit to a new hand that will receive another decision,
                # split to two hands that will each receive their own analysis,
                # or doubled to a hand that cannot receive any more decisions (busted or not)
                raise Exception("TODO")
            outcomes[action].append(outcome)

    return outcomes


def get_shoe_with_tc(target_tc, precision):
    tries = 0
    while True:
        shoe = get_6_deck_shoe()
        shoe.shuffle()
        while True:
            tc = shoe.get_true_count()
            if abs(tc - target_tc) <= precision:
                return shoe
            try:
                next(shoe)
            except StopIteration:
                break
        tries += 1
        if tries > 10000:
            raise Exception("could not find shoe with TC of {} +/- {}".format(target_tc, precision))


def get_evs_for_cell(player_hand, dealer_card):
    min_tc = -5
    max_tc = +5
    step = 1
    for _ in range(10):
        shoe = get_shoe_with_tc(target_tc, precision=step)


def get_all_evs():
    player_hands = get_possible_player_hands()
    dealer_cards = get_possible_dealer_cards()
    for player_hand in player_hands:
        for dealer_card in dealer_cards:
            # each cell in the chart
            evs = get_evs_for_cell(player_hand, dealer_card)
            print(evs)
            # once this actually works at all, graph them


if __name__ == "__main__":
    get_all_evs()
