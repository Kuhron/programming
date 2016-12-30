import Data.List


fizzBuzz :: Int -> [String]
fizzBuzz n = 
    let fizz x = if x `mod` 3 == 0 then "Fizz" else ""
        buzz x = if x `mod` 5 == 0 then "Buzz" else ""
        convertHelper x = fizz x ++ buzz x
        convert x = let c = convertHelper x in if c == "" then show x else c
        fizzBuzzNums = map convert [1 ..]
    in take n fizzBuzzNums


printFizzBuzz :: Int -> IO ()
printFizzBuzz n = putStr $ unlines $ fizzBuzz n