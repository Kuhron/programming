import PEUtil
import Data.List


main = do
    let n = 2000000
    let ps = takeWhile (< n) getPrimes

    -- memory hog!
    -- let result = foldl' (+) 0 $ ps
    -- putStrLn $ show $ result

    -- let x = last ps
    -- putStrLn $ show $ x

    let lastPrime = foldl' (\x y -> y) 0 $ ps
    putStrLn $ show $ lastPrime

    putStrLn "haven't fixed memory problem yet"
