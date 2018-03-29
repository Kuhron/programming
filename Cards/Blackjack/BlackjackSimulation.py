import argparse
import logging, logging.handlers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Cards.Blackjack.BlackjackCard import BlackjackCard as Card
from Cards.Blackjack.BlackjackHand import BlackjackHand as Hand
from Cards.Blackjack.Player import Player
from Cards.Blackjack.Dealer import Dealer
from Cards.Blackjack.Table import Table
from Cards.Blackjack.CountingAndBettingSystem import CountingAndBettingSystem


def add_card(hand, deck, is_face_up, counting_player):
    card = Card(next(deck), is_face_up)
    hand.add_card(card)
    logger.info("new card in hand {}".format(hand))
    if is_face_up:
        counting_player.count(card, deck)


def play_turn(player, table, dealer_card, counting_player):
    logger.info("-- playing turn for {}".format(player.name))
    shoe = table.shoe
    for hand in player.hands:
        while True:
            if hand.has_busted():
                logger.info("{} busted with hand {}".format(player, hand))
                break
            elif hand.is_blackjack():
                logger.info("{} has blackjack with hand {}".format(player, hand))
                break

            decision = player.decide(hand, dealer_card)

            # restrictions
            if decision == "D" and player.has_already_split() and not table.double_after_split:
                decision = "H"
            if decision == "P" and len(player.hands) >= table.max_hands_total:
                decision = "H"  # TODO is this always true? probably not (e.g. 8s with a high count); treat it as HARD_MATRIX rather than PAIR_MATRIX

            logger.info("{} has hand {} and decision {}".format(player, hand, decision))

            if decision == "H":
                add_card(hand, shoe, is_face_up=True, counting_player=counting_player)

            elif decision == "S":
                break

            elif decision == "D":
                player.double_bet(hand)
                # one card after double
                add_card(hand, shoe, is_face_up=True, counting_player=counting_player)
                break

            elif decision == "U":
                # hack up surrender as reducing bet to half and replacing hand with a busted one, so this bet will be lost
                # hopefully I don't implement counting in such a way as to screw it up here (player counting the bogus hand)
                QUEEN_OF_SPADES = Card.from_str("QS")
                player.halve_bet(hand)
                player.hands.remove(hand)
                player.hands.append(Hand([QUEEN_OF_SPADES] * 3))
                break

            elif decision == "P":
                assert hand in player.hands, "hand {} not in list:\n{}".format(hand, player.hands)
                # print(hand)
                # input()
                is_aces = hand.cards[0].value == hand.cards[1].value == "A"  # only one each after splitting aces, usually
                new_hands = hand.split()
                player.hands.remove(hand)
                for new_hand in new_hands:
                    add_card(new_hand, shoe, is_face_up=True, counting_player=counting_player)
                    player.hands.append(new_hand)
                if (not is_aces) or table.play_after_splitting_aces:
                    play_turn(player, table, dealer_card, counting_player)  # replay on the resulting hands
                break  # do not keep looping on old state (before split)
                # TODO add restrictions on play after split, including staying on non-splittable hand and splitting another one after that
            else:
                raise Exception("unhandled decision: {}".format(decision))


def reset_all_players(all_players, dealer):
    for pl in all_players:
        pl.reset()
    dealer.reset()


def play_round(player, table, with_other_players=True):
    # TODO: desperately needs to be broken down into smaller functions

    if with_other_players:
        n_other_players = np.random.choice([0, 1, 2, 3, 4, 5])
        other_players = [Player(table.minimum_bet * np.random.randint(1, 101), False, None) for i in range(n_other_players)]
    else:
        other_players = []
    all_players = other_players + [player]
    np.random.shuffle(all_players)  # mutates arg
    # DO NOT PUT DEALER IN all_players; treat them separately

    for i, pl in enumerate(all_players):
        pl.name = "main player" if pl is player else "other player {}".format(i)

    shoe = table.shoe
    dealer = table.dealer
    dealer.name = "dealer"

    # re-shuffle if necessary
    if shoe.is_dealt_out() or shoe.get_n_cards_left() < 30:
        table.shuffle_shoe()
        shoe = table.shoe  # table.shuffle_shoe() changes table.shoe to a different object, so re-assign this reference
        player.reset_count()
    logger.info("{:.2f} decks left in shoe".format(shoe.get_n_decks_left()))

    # initial bet
    for pl in all_players:
        pl.place_initial_bet(pl.hands[0], table)

    # initial deal
    for i in range(2):
        for pl in all_players:
            is_face_up = i == 1 or table.cards_face_up
            add_card(pl.hands[0], shoe, is_face_up, player)

        # dealer
        is_face_up = i == 1
        add_card(dealer.hands[0], shoe, is_face_up, player)

    is_face_up = True  # all cards that follow
    dealer_card = dealer.hands[0].cards[1]
    logger.info("dealer shows {}".format(dealer_card))

    if dealer_card.value == "A":
        for pl in all_players:
            if pl.will_take_insurance():
                assert len(pl.hands) == 1  # no one has had chance to split yet
                pl.insurance_bet = pl.hands[0].current_bet / 2  # half of original bet always, as far as I know
                logger.info("{} takes insurance, betting {:.2f}".format(pl, pl.insurance_bet))
                # raise; pl.bet(pl.hands[0], pl.insurance_bet)  # DON'T do this; the player will be overpaid if dealer has blackjack

    if dealer.has_blackjack():
        logger.info("dealer has blackjack")
        for pl in all_players:
            assert len(pl.hands) == 1  # no one has had chance to split yet
            hand = pl.hands[0]
            if (not dealer_card.value == "A") and hand.is_blackjack():
                # assume player declared blackjack immediately or cards are face up (some games will only pay even money otherwise)
                pl.win_on_hand(hand.current_bet * (1 + table.blackjack_payoff_ratio))
            # elif dealer_card.value == "A" and pl.has_insurance():  # redundant; can't take insurance unless dealer shows ace
            elif pl.has_insurance():
                pl.win_on_hand(pl.insurance_bet * (table.insurance_payoff_ratio))
                # note do not win back current bet (it is lost since dealer has blackjack)
            else:
                pl.lose_on_hand()
        reset_all_players(all_players, dealer)
        return
    elif dealer_card.value == "A":
        logger.info("dealer does not have blackjack")

    for pl in all_players:
        if pl.has_insurance():
            pl.lose_insurance_bet()

    # player turns
    for pl in all_players:
        play_turn(pl, table, dealer_card, player)

    # dealer turn
    # show cards
    for card in dealer.hands[0].cards:
        if not card.is_face_up:
            logger.info("dealer flipped over {}".format(card))
            player.count(card, shoe)
            card.is_face_up = True
    play_turn(dealer, table, None, player)

    dealer_hand_value = dealer.hands[0].max_value

    # payoffs
    for pl in all_players:
        for hand in pl.hands:
            if hand.has_busted():  # regardless of dealer outcome
                pl.lose_on_hand()
            elif hand.is_blackjack() and (len(pl.hands) == 1 or table.pay_blackjack_after_split):
                # remember, always show blackjack immediately if you are dealt it! (some tables will only pay even money otherwise)
                pl.win_on_hand(hand.current_bet * (1 + table.blackjack_payoff_ratio))
            elif dealer.hands[0].has_busted():
                pl.win_on_hand(hand.current_bet * 2)
            elif hand.max_value > dealer_hand_value:
                pl.win_on_hand(hand.current_bet * 2)
            elif hand.max_value == dealer_hand_value:
                # push
                pl.win_on_hand(hand.current_bet)
            else:
                pl.lose_on_hand()

    # count remaining cards
    for pl in all_players:
        for hand in pl.hands:
            for card in hand.cards:
                if not card.is_face_up:
                    card.is_face_up = True  # pointless, but for consistency
                    logger.info("{} flipped over {}".format(pl, card))
                    player.count(card, shoe)

    # reset everyone
    reset_all_players(all_players, dealer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="n_rounds", type=int, default=1000)
    parser.add_argument("-v", dest="verbose", action="store_true")
    args = parser.parse_args()

    # logging setup
    log_fh = logging.FileHandler('BJS.log', mode="w")
    log_fh.setLevel(logging.INFO)
    log_fh.setFormatter(logging.Formatter('%(message)s'))

    root = logging.getLogger()
    if args.verbose:
        root.setLevel(logging.INFO)
        root.addHandler(log_fh)

    logger = logging.getLogger(__name__)

    # game parameters
    table = Table(
        doubleable_hard_values = [x for x in range(2, 21)], #[9, 10, 11],  # (orig. [10, 11])
        minimum_bet = 1,  # (orig. 5), but make it 1 for easier EV idea
        maximum_bet = 1000,  # this may be the biggest factor limiting gains when the count is high (orig. 200)
        blackjack_payoff_ratio = 3/2,  # (orig. 3/2)
        insurance_payoff_ratio = 2/1,  # (orig. 2/1)
        n_decks = 6,
        max_hands_total = 4,  # limit splitting
        double_after_split = True,
        hit_more_than_once_after_split = False,
        cards_face_up = True,
        stay_on_soft_17 = True,
        pay_blackjack_after_split = False,  # usually False, tables treat this as normal 21
        play_after_splitting_aces = False,  # usually False, splitting aces will typically give one card each with no further play
    )

    count_function_of_value = CountingAndBettingSystem.hi_lo_count_function_of_value

    def bet_function_of_tc(tc):
        threshold = 0.01
        bet_ratio = 5  # number of minimum bets that the player will increase bet by for each additional TC unit
        def transform(tc):
            # return 0  # always bet table minimum
            # return np.random.pareto(1 + 1/tc)
            return tc * bet_ratio
            # return tc ** 0.5
            # return np.log(tc)
            # return tc**4
            # return np.inf  # will bet table max
            # return np.piecewise(tc, [tc < 3, 3 <= tc <= 5, 5 < tc], [0, (100 - 0) * (tc - 3)/(5 - 3), 100])
        return table.minimum_bet * transform(tc) if tc >= threshold else 0

    counting_and_betting_system = CountingAndBettingSystem(count_function_of_value, bet_function_of_tc)

    initial_bankroll = 100000 * table.minimum_bet
    player = Player(initial_bankroll, is_counting=True, counting_and_betting_system=counting_and_betting_system)
    # giving player so much money prevents typical drawdowns from bankrupting them

    bankrolls = [player.bankroll]
    counts = [0]
    true_counts = [0]
    n_rounds = 0
    while True:
        logger.info("\n---- round {} ----".format(n_rounds))
        if n_rounds > args.n_rounds:
            break
        play_round(player, table, with_other_players=True)
        bankrolls.append(player.bankroll)
        counts.append(player.running_count)
        true_counts.append(player.true_count)
        if player.is_broke():
            break
        n_rounds += 1

    pnls = [x - initial_bankroll for x in bankrolls]

    d_cash = np.diff(np.array(bankrolls))
    ev = np.mean(d_cash)
    sd = np.std(d_cash)
    print("\n\nEV {:.2f} , SD {:.2f}".format(ev, sd))

    # plt.plot(counts, c="r")
    # plt.plot(true_counts, c="r")

    # ax2 = plt.gca().twinx()
    plt.plot(pnls, c="g")
    plt.show()

    # plt.hist(d_cash, bins=50)
    # plt.show()

    # TODO: be able to reproduce the statistics table at https://wizardofodds.com/games/blackjack/card-counting/high-low/
    # TODO: implement insurance when TC > +3 (players are not yet betting on it even though I put this in the function)
