module PEUtil where


import qualified Data.Sequence as Seq
import Data.Foldable (toList)


type Primes = [Int]
type PrimeFlags = Seq.Seq Bool


getPrimes :: Primes
getPrimes = map (head . fst) getPrimeTuples


getPrimeTuples :: [(Primes, PrimeFlags)]
getPrimeTuples =
    let flags = repeat True
        primeFlags = Seq.fromList flags
        initSieve = ([2], primeFlags)
    in iterate iterSieve initSieve


iterSieve :: (Primes, PrimeFlags) -> (Primes, PrimeFlags)
iterSieve (knownPrimes, others) =
    let largestKnownPrime = head knownPrimes
        p = fst . head $ filter (\(x, y) -> and [x > largestKnownPrime, y]) $ zip [2 ..] (toList others)
        news = crossOff p others
    in (p : knownPrimes, news)


crossOff :: Int -> PrimeFlags -> PrimeFlags
crossOff p primeFlags =
    let mults = map (* p) [p ..]
        crossOffMult flags mult = Seq.update (mult - 2) False flags
        newFlags = foldl crossOffMult primeFlags mults
    in newFlags


getNthPrime :: Int -> Int
getNthPrime n = 
    let primes = getPrimes
    in primes !! (n - 1)
