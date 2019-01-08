import PEUtil


main = do
    let n = 2000000
    let ps = takeWhile (< n) getPrimes
    let result = sum ps
    putStrLn $ show $ result