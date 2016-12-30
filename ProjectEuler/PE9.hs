import qualified Data.List as L


getIntTripletsWithSum :: Int -> [[Int]]
getIntTripletsWithSum n =
    let a = [[x, y] | x <- [1 .. n], y <- [1 .. n]]
        rawTrips = map (\lst -> (n - (sum lst)) : lst) a
        sortedTrips = map L.sort rawTrips
        allNatural = all (> 0)
    in filter allNatural sortedTrips


pythagorean :: [Int] -> Bool
pythagorean [a, b, c] = a ^ 2 + b ^ 2 == c ^ 2


main = do
    let trips = getIntTripletsWithSum 1000
    let trip = head $ L.nub $ filter pythagorean trips  -- due to inefficiency of nub, do it as late as possible (after filtering)
    let result = product trip
    putStrLn $ show $ result