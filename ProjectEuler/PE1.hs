getSumOfMultiplesBelowWithDuplicates :: (Integral a) => [a] -> a -> a
getSumOfMultiplesBelowWithDuplicates xs n =
    let multipliers = fmap (*) [1 ..]
        multiples x = takeWhile (< n) $ multipliers <*> [x]
    in sum $ xs >>= multiples


getSumOfMultiplesBelow :: (Integral a) => [a] -> a -> a
getSumOfMultiplesBelow xs n =
    let isMultiple y x = y `mod` x == 0
        isAnyMultiple xs y = any (isMultiple y) xs
        multiples = filter (isAnyMultiple xs) [1 .. n-1]
    in sum multiples


main = do
    let s = [3, 5] `getSumOfMultiplesBelow` 1000
    putStrLn $ show s