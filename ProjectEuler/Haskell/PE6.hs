sumSquare :: (Integral a) => Int -> a
sumSquare n =
    let squares = take n $ map (^ 2) [1 ..]
    in sum squares

squareSum :: (Integral a) => Int -> a
squareSum n = 
    let s = sum $ take n [1 ..]
    in s ^ 2

diff :: (Integral a) => Int -> a
diff n =
    let a = sumSquare n
        b = squareSum n
    in abs $ a - b


main = do
    let n = 100
    let result = diff n
    putStrLn $ show $ result