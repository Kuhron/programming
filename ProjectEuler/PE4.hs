isPalindrome :: (Show a) => a -> Bool
isPalindrome x =
    show x == (reverse . show $ x)

getProds :: (Integral a) => a -> [a]
getProds n =
    (*) <$> (reverse [1 .. n]) <*> (reverse [1 .. n])

main = do
    let result = maximum $ filter isPalindrome $ getProds 999
    putStrLn $ show $ result