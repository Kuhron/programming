import unittest

import BlackjackSimulation as bjs


class GetHandValuesTest(unittest.TestCase):
    def test_blackjack(self):
        h = bjs.Hand.from_str_list(["AS", "JS"])
        self.assertEqual(h.get_values(), (11, 21))
        h = bjs.Hand.from_str_list(["AS", "QS"])
        self.assertEqual(h.get_values(), (11, 21))
        h = bjs.Hand.from_str_list(["AS", "KS"])
        self.assertEqual(h.get_values(), (11, 21))
        h = bjs.Hand.from_str_list(["AS", "TS"])
        self.assertEqual(h.get_values(), (11, 21))

    def test_hard_values(self):
        h = bjs.Hand.from_str_list(["KS", "JS"])
        self.assertEqual(h.get_values(), (20, None))
        h = bjs.Hand.from_str_list(["5S", "9S"])
        self.assertEqual(h.get_values(), (14, None))

    def test_soft_values(self):
        h = bjs.Hand.from_str_list(["AS", "5S"])
        self.assertEqual(h.get_values(), (6, 16))
        h = bjs.Hand.from_str_list(["AS", "2S"])
        self.assertEqual(h.get_values(), (3, 13))
        h = bjs.Hand.from_str_list(["AS", "AS"])
        self.assertEqual(h.get_values(), (2, 12))

    def test_busted_soft_values(self):
        h = bjs.Hand.from_str_list(["AS", "KS", "8S"])
        self.assertEqual(h.get_values(), (19, None))
        h = bjs.Hand.from_str_list(["AS"] * 21)
        self.assertEqual(h.get_values(), (21, None))


class GetBasicStrategyActionTest(unittest.TestCase):
    def test_hard_hands(self):
        h = bjs.Hand.from_str_list(["AS", "KS", "8S"])
        dc = bjs.BlackjackCard.from_str("5S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "S")
        h = bjs.Hand.from_str_list(["2S", "KS", "4S"])
        dc = bjs.BlackjackCard.from_str("TS")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "U")
        h = bjs.Hand.from_str_list(["4S", "8S"])
        dc = bjs.BlackjackCard.from_str("8S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "H")
        h = bjs.Hand.from_str_list(["6S", "4S"])
        dc = bjs.BlackjackCard.from_str("2S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "D")

    def test_soft_hands(self):
        h = bjs.Hand.from_str_list(["AS", "KS"])
        dc = bjs.BlackjackCard.from_str("5S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "S")
        h = bjs.Hand.from_str_list(["AS", "7S"])
        dc = bjs.BlackjackCard.from_str("TS")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "H")  # most misplayed hand in blackjack
        h = bjs.Hand.from_str_list(["AS", "4S"])
        dc = bjs.BlackjackCard.from_str("5S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "D")

    def test_pair_hands(self):
        h = bjs.Hand.from_str_list(["8S", "8S"])
        dc = bjs.BlackjackCard.from_str("5S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "P")
        h = bjs.Hand.from_str_list(["AS", "AS"])
        dc = bjs.BlackjackCard.from_str("5S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "P")
        h = bjs.Hand.from_str_list(["5S", "5S"])
        dc = bjs.BlackjackCard.from_str("5S")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "D")
        h = bjs.Hand.from_str_list(["3S", "3S"])
        dc = bjs.BlackjackCard.from_str("AS")
        self.assertEqual(bjs.BasicStrategy.get_action(h, dc), "H")


if __name__ == "__main__":
    unittest.main()