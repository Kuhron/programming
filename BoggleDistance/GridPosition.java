import java.util.ArrayList;


public class GridPosition {
    int row;
    int col;
    int[] position = new int[2];

    public GridPosition(int row, int col) {
        this.position = new int[] {row, col};
        this.row = row;
        this.col = col;
    }

    public ArrayList<GridPosition> getNeighbors(int gridSize) {
        ArrayList<GridPosition> res = new ArrayList<GridPosition>();
        int row = this.position[0];
        int col = this.position[1];
        int minRow = Math.max(0, row-1);
        int maxRow = Math.min(gridSize-1, row+1);
        int minCol = Math.max(0, col-1);
        int maxCol = Math.min(gridSize-1, col+1);

        for (int i = minRow; i < maxRow+1; i++) {
            for (int j = minCol; j < maxCol+1; j++) {
                boolean isSameAsStart = (i == row && j == col);
                if (!isSameAsStart) {
                    GridPosition neighbor = new GridPosition(i, j);
                    res.add(neighbor);
                }
            }
        }
        // System.out.printf("the neighbors of %s are %s\n", this, res);
        return res;
    }

    public boolean isAllowed(AllowedGrid allowed) {
        return allowed.allows(this);
    }

    public String toString() {
        return String.format("[%d, %d]", this.row, this.col);
    }
}
