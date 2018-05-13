import random


class CrystalGrowthRules:
    original_rules = [
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1],
            ],
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
            ],
            [
                [1, 0, 1],
                [0, 0, 0],
                [0, 1, 0],
            ],
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
            ],
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ],
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 0],
            ],
            [
                [0, 1, 1],
                [0, 0, 0],
                [1, 1, 0],
            ],
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 1],
            ],
            [
                [0, 1, 0],
                [0, 0, 0],
                [1, 0, 1],
            ],
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1],
            ],
            [
                [0, 1, 0],
                [0, 0, 0],
                [1, 1, 1],
            ],
        ],
        [
            [
                [0, 0, 0],
                [1, 0, 1],
                [1, 0, 1],
            ],
            [
                [0, 1, 0],
                [1, 0, 1],
                [1, 0, 1],
            ],
        ],
        [
            [
                [0, 0, 0],
                [1, 0, 1],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
        ],
        [
            [
                [0, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
        ],
    ]

    diamond_rules = original_rules[:-1]

    test_directionality_rules = [  # should just grow up in a straight line, but making sure I am not mixing up row/column somewhere
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ],
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
            ],
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ],
        ],
    ]

    @staticmethod
    def generate_random_array():
        return [[random.choice([0, 1]) for j in range(3)] for i in range(3)]

    @staticmethod
    def generate_random_rules():
        result = [
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                CrystalGrowthRules.generate_random_array()
            ]
        ]
        while len(result) < 20 or random.random() < 0.7:
            result.append([CrystalGrowthRules.generate_random_array() for x in range(2)])
        return result
