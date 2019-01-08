--import qualified Data.Set as S

----sortedListIntersection :: [a] -> [a]
----sortedListIntersection (x:xs) (y:ys) = 

--multiIntersection :: (Ord a) => [S.Set a] -> S.Set a
--multiIntersection [x] = x
--multiIntersection (x:xs) = S.intersection x $ multiIntersection xs

--multiples :: (Integral a) => a -> [a]
--multiples n = 
--    let big_n = 1000000  -- just make it bigger if the set is empty at the end
--    in map (* n) [1 .. big_n]

--main = do
--    let n = 20
--    let set = multiIntersection $ map (S.fromList . multiples) [1 .. n]
--    let result = minimum set
--    putStrLn $ show $ result


--divisibleFuncs :: (Integral a) => a -> [a -> Bool]
--divisibleFuncs n = map (\x -> \y -> y `mod` x == 0) [1 .. n]

--divisible :: (Integral a) => a -> a -> Bool
--divisible n x =
--    let fs = divisibleFuncs n
--    in and $ map ($ x) fs

--main = do
--    let n = 20
--    let f = divisible n
--    let result = head $ filter f [1 ..]
--    putStrLn $ show $ result

factors :: (Integral a) => a -> [a]
factors 1 = []
factors n = 
    let facs = factors $ n - 1
    in facs ++ [getFactor facs n]

getFactor :: (Integral a) => [a] -> a -> a
getFactor facs n
    | length facs == 0         = n
    | n `mod` (head facs) == 0 = getFactor (tail facs) (n `div` (head facs))
    | otherwise                = getFactor (tail facs) n

lcm' :: (Integral a) => a -> a
lcm' n =
    let facs = factors n
    in foldl (*) 1 facs


main = do
    let n = 20
    let result = lcm' n
    putStrLn $ show $ result