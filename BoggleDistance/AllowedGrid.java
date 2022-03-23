public class AllowedGrid {
    private boolean[][] grid;
    int size;

    public AllowedGrid(int size, boolean[][] grid) {
        this.size = size;
        this.grid = grid;
    }

    public static AllowedGrid allTrue(int size) {
        boolean[][] grid = new boolean[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                grid[i][j] = true;
            }
        }
        return new AllowedGrid(size, grid);
    }

    public AllowedGrid withChange(GridPosition index, boolean newValue) {
        boolean[][] newGrid = new boolean[size][size];
        int row = index.row;
        int col = index.col;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (row == i && col == j) {
                    newGrid[i][j] = newValue;
                } else {
                    newGrid[i][j] = grid[i][j];
                }
            }
        }
        return new AllowedGrid(size, newGrid);
    }

    public boolean allows(GridPosition index) {
        int row = index.row;
        int col = index.col;
        return grid[row][col];
    }
}
