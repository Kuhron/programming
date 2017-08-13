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
        h = bjs.Hand.from_str_list(["AS", "AS"])
        self.assertEqual(h.get_values(), (2, 12))

    def test_soft_values(self):
        h = bjs.Hand.from_str_list(["AS", "5S"])
        self.assertEqual(h.get_values(), (6, 16))
        h = bjs.Hand.from_str_list(["AS", "2S"])
        self.assertEqual(h.get_values(), (3, 13))

if __name__ == "__main__":
    unittest.main()