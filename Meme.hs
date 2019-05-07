data Meme = Meme String
instance Show Meme where
    show (Meme m) = m ++ " Memes"

data Teen = Teen String
instance Show Teen where
    show (Teen t) = t ++ " Teens"

data Group = Group String String
instance Show Group where
    show (Group m t) = (show (Meme m)) ++ " for " ++ (show (Teen t))

getGroup = do
    putStrLn "Meme descriptor:"
    meme <- getLine
    putStrLn "Teen descriptor:"
    teen <- getLine
    let g = Group meme teen
    return g

main = do
    g <- getGroup
    putStrLn ""
    print g

