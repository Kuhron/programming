-- to see if takewhile is stopping once the condition fails or checking all of them for the condition
-- result: it stops the first time the condition fails, so this code just prints [1,2]

lst = map (\x -> mod x 4) [1 ..]
l2 = takeWhile (< 3) lst

main = do
    putStrLn $ show $ l2

