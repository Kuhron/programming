public class BoggleDistance {
    public static void main(String[] args) {
        int size = 4;
        BoggleGrid grid = BoggleGrid.random(size);
        grid.print();
        // int[][] paths = grid.getAllPaths();
        grid.printAllStrings();

        // TODO enumerate all possible grid paths
        // TODO trie structure of words in cmudict
        // TODO find all words in grid
        // TODO play with metrics of how similar certain paths are, find the outlying words (whose paths are not very similar to other words in the grid)
    }
}
