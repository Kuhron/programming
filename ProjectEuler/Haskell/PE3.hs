primeFactors :: (Integral a) => a -> [a]
primeFactors n = let
    initPrimes = [2]
    iterations = iterate iterPrimeFactors (n, initPrimes)
    done (x, _) = x == 1
    (_, primes) = head $ filter done iterations
    in primes


iterPrimeFactors :: (Integral a) => (a, [a]) -> (a, [a])
iterPrimeFactors (n, primes)
    | n == 1 = (n, primes)
    | n `mod` (head primes) == 0 = (n `div` (head primes), primes)
    | otherwise = let
        p = nextPrime primes
        in iterPrimeFactors (n, p : primes)


nextPrime :: (Integral a) => [a] -> a
nextPrime primes = let
    q = maximum primes + 1
    coprime x y = and [(x `mod` y /= 0), (y `mod` x /= 0)]
    coprimeFuncList = map coprime primes
    coprimeAll n = and $ coprimeFuncList <*> [n]
    candidates = filter coprimeAll [q ..]
    in head candidates


main = do
    let n = 600851475143
    let result = maximum $ primeFactors n
    putStrLn $ show $ result