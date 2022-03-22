import java.util.Random;
import java.util.ArrayList;


public class BoggleGrid {
    int size;  // how many rows and columns are in the grid
    char[][] grid;

    public BoggleGrid(int size, char[][] grid) {
        this.size = size;
        this.grid = grid;
    }

    public static BoggleGrid random(int size) {
        Random r = new Random();
        // https://stackoverflow.com/questions/2626835/is-there-functionality-to-generate-a-random-character-in-java
        char[][] grid = new char[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                char c = (char)(r.nextInt(26) + 'A');
                grid[i][j] = c;
            }
        }
        return new BoggleGrid(size, grid);
    }

    public void print() {
        System.out.println("<BoggleGrid");
        for (int i = 0; i < size; i++) {
            String row = "    ";
            for (int j = 0; j < size; j++) {
                char c = grid[i][j];
                String s = Character.toString(c);
                row = row + s + " ";
            }
            System.out.println(row);
        }
        System.out.println(">");
    }

    public char charAt(int[] index) {
        assert index.length == 2;
        int i = index[0];
        int j = index[1];
        return grid[i][j];
    }

    // public PathTree getAllPaths() {
    //     PathTree paths = new PathTree();
    //     // int nPaths = paths.length;
    //     // System.out.printf("got %d new paths\n", nPaths);
    //     return paths;
    // }

    // public int[][] getAllPathsFromStartingPoint(int[] startingPoint, boolean[][] allowed) {

    // }

    // public int[][] getNextStepsFromStartingPoint(int[] startingPoint, boolean[][] allowed) {
    //     // allowed is a grid that says whether each cell can be used or not, set them to false in recursive calls when you've used a letter

    // }

    public void printAllStrings() {
        System.out.println("printing all strings of this BoggleGrid, regardless of if they are words");
        boolean[][] allowed = new boolean[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                allowed[i][j] = true;
            }
        }

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                GridPosition start = new GridPosition(i, j);
                printStringsFromInitialCondition(start, allowed);
            }
        }
    }

    public void printStringsFromInitialCondition(GridPosition start, boolean[][] allowed) {
        // starting at a certain point in the grid, with certain cells allowed to be used, print the strings you can get
        System.out.println("not implemented");
    }

    public ArrayList<GridPosition> getPossibleNextPositions(GridPosition position, boolean[][] allowed) {
        ArrayList<GridPosition> neighbors = position.getNeighbors(this.size);
        ArrayList<GridPosition> options = new ArrayList<GridPosition>();
        for (GridPosition neighbor : neighbors) {
            if (neighbor.isAllowed(allowed)) {
                options.add(neighbor);
            }
        }
        return options;
    }
}
