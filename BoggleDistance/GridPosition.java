import java.util.ArrayList;


public class GridPosition {
    int[] position = new int[2];

    public GridPosition(int row, int col) {
        this.position = new int[] {row, col};
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
                if (i != j) {
                    GridPosition neighbor = new GridPosition(i, j);
                    res.add(neighbor);
                }
            }
        }
        return res;
    }

    public boolean isAllowed(boolean[][] allowed) {
        return allowed[this.position[0]][this.position[1]];
    }
}
