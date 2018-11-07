rule_str = """
Object of game is to build as much "structure" as you can
before entropy wins out. Structures are of various types,
such as 2 5 8 J, all diamonds, 7 of each suit, etc.
(need these to be more clearly defined for scoring)
There are 12 card values that use modular arithmetic.
K is 0/12, A is 1, and J is 11.
Queens are all agents of entropy,
and the Queen of Spades is Eris herself.
Once Eris appears, the game ends and no further structure
can be built.
Score is based on structure points - entropy created.
Structure should be defined based on probability of the hand.
Entropy is number of cards seen.
"""


