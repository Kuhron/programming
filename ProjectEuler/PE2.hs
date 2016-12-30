--fibonacci :: (Integral a) => a -> a
--fibonacci n
--    | n <= 1    = 1
--    | otherwise = fibonacci (n - 2) + fibonacci (n - 1)


--getFibonaccis :: (Integral a) => a -> [a]
--getFibonaccis n
--    | n == 0 = [1]
--    | n == 1 = [1, 1]
--    | otherwise = let
--        lst = getFibonaccis (n - 1)


--getFibonaccis :: (Integral a) => [a]
--getFibonaccis = let
--    f x1 x2 = x1 : x2 : (x1 + x2)
--    in 1 : 1 : getFibonaccis


--extendFibs :: (Integral a) => [a] -> [a]
--extendFibs xs
--    | length xs < 2 = 1 : xs
--    | otherwise     = let
--        x1 = xs !! (length xs)
--        x2 = xs !! (length xs - 2)
--        in xs ++ [x1 + x2]


getFibTuple :: (Integral a) => (a, a, a) -> (a, a, a)
getFibTuple (_, x1, x2) = (x1, x2, x1 + x2)


getFibTuples :: (Integral a) => [(a, a, a)]
getFibTuples = iterate getFibTuple (0, 0, 1)


extract :: (Integral a) => (a, a, a) -> a
extract (_, _, x) = x

getFibonaccis :: (Integral a) => [a]
getFibonaccis = map extract getFibTuples


main = do
    let n = 4000000
    let fibs = takeWhile (<= n) $ filter even $ getFibonaccis
    putStrLn $ show $ sum fibs