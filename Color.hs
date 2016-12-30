module Color where


data Color a = Red a | Yellow a | Blue a deriving (Show, Read)


instance Functor Color where
    fmap f (Red x) = Red (f x)
    fmap f (Yellow x) = Yellow (f x)
    fmap f (Blue x) = Blue (f x)


instance (Num t, Fractional t) => Num (Color t) where
    Red a + Red b = Red (a + b)
    Yellow a + Yellow b = Yellow (a + b)
    Blue a + Blue b = Blue (a + b)

    Red a + Yellow b = Blue (a - b)
    Yellow a + Blue b = Red (a - b)
    Blue a + Red b = Yellow (a - b)

    Red a + Blue b = Yellow (b - a)
    Blue a + Yellow b = Red (b - a)
    Yellow a + Red b = Blue (b - a)

    Red a * Red b = Red (a * b)
    Yellow a * Yellow b = Yellow (a * b)
    Blue a * Blue b = Blue (a * b)

    Red a * Yellow b = Blue (a / b)
    Yellow a * Blue b = Red (a / b)
    Blue a * Red b = Yellow (a / b)

    Red a * Blue b = Yellow (b / a)
    Blue a * Yellow b = Red (b / a)
    Yellow a * Red b = Blue (b / a)

    abs (Red a) = Red (abs a)
    abs (Yellow a) = Yellow (abs a)
    abs (Blue a) = Blue (abs a)

    signum (Red a) = Red (signum a)
    signum (Yellow a) = Yellow (signum a)
    signum (Blue a) = Blue (signum a)

    negate (Red a) = Red (negate a)
    negate (Yellow a) = Yellow (negate a)
    negate (Blue a) = Blue (negate a)

    --fromInteger a
    --    | a `mod` 3 == 0 = Red a
    --    | a `mod` 3 == 1 = Yellow a
    --    | a `mod` 3 == 2 = Blue a


instance (Eq t) => Eq (Color t) where
    Red a == Red b = a == b
    Yellow a == Yellow b = a == b
    Blue a == Blue b = a == b
    Red a == Yellow b = False
    Red a == Blue b = False
    Yellow a == Red b = False
    Yellow a == Blue b = False
    Blue a == Red b = False
    Blue a == Yellow b = False

    -- there has got to be a better way to do this. I want to basically write:
    -- color a == color b = a == b
    -- color a == otherColor b = False


instance (Ord t) => Ord (Color t) where
    -- color a `compare` color b = a `compare` b
    Red a `compare` Red b = a `compare` b
    Yellow a `compare` Yellow b = a `compare` b
    Blue a `compare` Blue b = a `compare` b
    Red a `compare` Yellow b = GT
    Yellow a `compare` Blue b = GT
    Blue a `compare` Red b = GT

    -- similarly here, I want to make the type constructors (Red, Yellow, Blue) have an ordering, so I can define:
    -- color1 a `compare` color2 b = color1 `compare` color2