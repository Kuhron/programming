public class PathTree<T> {
    // tree structure that stores which paths between positions are possible in the BoggleGrid
    // save space vs enumerating every path as a separate array

    private PathTreeNode<T> root;
    // in the top of the tree where you didn't come from anything and you could start with various starting points, the root is null

    public PathTree<T>() {
        this.root = null;
    }

    public void add(Path<T> path) {
        throw
    }

    public int getNumberOfPaths() {
        System.out.println("not implemented");
        return null;
    }
}

public class PathTreeNode<T> {
    T current;
    PathTreeNode<T>[] nextNodes;

    public PathTreeNode<T>(T current, PathTreeNode<T>[] nextNodes) {
        this.current = current;
        this.nextNodes = nextNodes;
    }
}

