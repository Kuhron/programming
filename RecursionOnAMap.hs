import Data.List

expandCoords :: (Int, Int) -> (Int, Int)
expandCoords (x, y) = (2*x, 2*y)

expandTriangle :: [((Int, Int), Int)] -> Int -> [((Int, Int), Int)]
expandTriangle initialTriangle nIterations = let
    iterationNumbers = range1 nIterations
    -- newTriangle = (iterate expandTriangleOnce initialTriangle) !! nIterations
    newTriangle = foldl expandTriangleOnce initialTriangle iterationNumbers
    in newTriangle

expandTriangleOnce :: [((Int, Int), Int)] -> Int -> [((Int, Int), Int)]
expandTriangleOnce triangle iterationNumber = let
    -- expand the triangle by a factor of 2, add in new nodes
    -- need iteration number to know the cyclic polarity of the child values
    currentCoords = getCoordPairsFromTriangle triangle
    newCoords = map expandCoords currentCoords
    newChildrenCoords = map getChildrenCoords newCoords  -- coords should already be expanded before inserting children
    currentValues = getValuesFromTriangle triangle
    newChildrenValues = zipWith getChildValues currentValues (replicate (length currentValues) iterationNumber)
    leftChildrenCoords = map fst newChildrenCoords
    leftChildrenValues = map fst newChildrenValues
    rightChildrenCoords = map snd newChildrenCoords
    rightChildrenValues = map snd newChildrenValues
    expandedOriginalTriangle = [(newCoords !! i , currentValues !! i) | i <- range1 (length currentValues)]
    leftAdditions = [(leftChildrenCoords !! i , leftChildrenValues !! i) | i <- range1 (length leftChildrenValues)]
    rightAdditions = [(rightChildrenCoords !! i , rightChildrenValues !! i) | i <- range1 (length rightChildrenValues)]
    newAdditionsToTriangle = leftAdditions ++ rightAdditions
    result = expandedOriginalTriangle ++ newAdditionsToTriangle
    in result

getChildrenCoords :: (Int, Int) -> ((Int, Int), (Int, Int))
getChildrenCoords (row, col) = ((row+1, col-1), (row+1, col+1))

getChildValues :: Int -> Int -> (Int, Int)
getChildValues parentValue iterationNumber = let
    flip = iterationNumber `mod` 2
    result = case parentValue of
        1 -> case flip of
            0 -> (2, 3)
            1 -> (3, 2)
        2 -> case flip of 
            0 -> (3, 1)
            1 -> (1, 3)
        3 -> case flip of
            0 -> (1, 2)
            1 -> (2, 1)
    in result

convertTriangleDataStructureToListOfLists :: [((Int, Int), Int)] -> [[Int]]
convertTriangleDataStructureToListOfLists triangle = let
    coordPairs = getCoordPairsFromTriangle triangle
    values = getValuesFromTriangle triangle
    rowIndices = getRowNumbersFromCoordPairs coordPairs
    colIndices = getColNumbersFromCoordPairs coordPairs
    maxRow = maximum rowIndices
    maxCol = maximum colIndices
    allRowNums = [0 .. maxRow]
    allColNums = [0 .. maxCol]
    -- row and col count from zero
    --don't use: allCoords = [(x, y) | x <- [0 .. maxRow], y <- [0 .. maxCol]]
    --don't use: hasValue = map (`elem` coordPairs) allCoords
    startingValues = [[0 | col <- allColNums] | row <- allRowNums]
    result = replaceMultipleValues2D startingValues rowIndices colIndices values
    in result

getCoordPairsFromTriangle :: [((a,a), a)] -> [(a,a)]
getCoordPairsFromTriangle triangle = map fst triangle
-- can't use !! indexing on tuples, function "fst" gets first element of 2-tuple, "snd" gets second one
-- for getting other indexes of longer tuples will need to write an auxiliary function

getValuesFromTriangle :: [((a,a), a)] -> [a]
getValuesFromTriangle triangle = map snd triangle

getRowNumbersFromCoordPairs :: [(a,a)] -> [a]
getRowNumbersFromCoordPairs pairs = map fst pairs

getColNumbersFromCoordPairs :: [(a,a)] -> [a]
getColNumbersFromCoordPairs pairs = map snd pairs

nRows2D :: [[a]] -> Int
nRows2D lst = length lst

nCols2D :: [[a]] -> Int
nCols2D lst 
    | length lst == 0 = 0
    | otherwise = length (lst !! 0)

replaceValue2D :: [[a]] -> Int -> Int -> a -> [[a]]
replaceValue2D lst rowIndex colIndex newValue = let
    nRows = nRows2D lst
    nCols = nCols2D lst
    beforeRows = [lst !! x | x <- range1 rowIndex]
    afterRows = [lst !! x | x <- range2 (rowIndex + 1) nRows]
    row = lst !! rowIndex
    beforeCols = [row !! x | x <- range1 colIndex]
    afterCols = [row !! x | x <- range2 (colIndex + 1) nCols]
    elementToChange = row !! colIndex
    newRow = beforeCols ++ [newValue] ++ afterCols
    newLst = beforeRows ++ [newRow] ++ afterRows
    in newLst

replaceValue2DGivenListsAndIndex :: [[a]] -> [Int] -> [Int] -> [a] -> Int -> [[a]]
replaceValue2DGivenListsAndIndex lst rowIndices colIndices newValues valueIndex = let
    thisRowIndex = rowIndices !! valueIndex
    thisColIndex = colIndices !! valueIndex
    newValue = newValues !! valueIndex
    in replaceValue2D lst thisRowIndex thisColIndex newValue

replaceMultipleValues2D :: [[a]] -> [Int] -> [Int] -> [a] -> [[a]]
replaceMultipleValues2D lst rowIndices colIndices newValues = let
    func lastList valueIndex = replaceValue2DGivenListsAndIndex lastList rowIndices colIndices newValues valueIndex
    nValues = length newValues
    valueIndices = range1 nValues
    foldingValueList = valueIndices -- will go to each int from 0 to number of times we're changing an element, each new index tells us which row/column/value to go get in order to make the next change to the array
    initialValue = lst
    in foldl func initialValue foldingValueList

range1 :: Int -> [Int]
range1 n = [0 .. n-1]

range2 :: Int -> Int -> [Int]
range2 m n = [m .. n-1]

convertIntToMapLetters n = case n of
    0 -> " "
    1 -> "M"
    2 -> "A"
    3 -> "P"
    _ -> " "

convert2DListToMapLetters = (map . map) convertIntToMapLetters

printFromNumberTriangle :: [((Int,Int), Int)] -> IO ()
printFromNumberTriangle triangle = let
    lst = convertTriangleDataStructureToListOfLists triangle
    strings2D = convert2DListToMapLetters lst
    strings1D = map unwords strings2D
    printable = intercalate "\n" strings1D
    in putStrLn printable


main = do
    let initialTriangle = [((0, 1), 1)]
    let newTriangle = expandTriangle initialTriangle 5
    -- print newTriangle
    -- print "---"
    -- print $ convertTriangleDataStructureToListOfLists newTriangle
    -- print "---"
    printFromNumberTriangle newTriangle
