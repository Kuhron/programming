public class AdjacencyListUndirectedGraph<T> {
    // idea for now: implement list of nodes as SinglyLinkedList, and also each node's neighbors as SinglyLinkedList
    // drawback that finding the neighbors of a node is O(n) to get the list for that node in the first place, but I don't want to do hash tables right now

    private SinglyLinkedList<Node<T>> nodes;

    private class Node<T> {
        T data;
        SinglyLinkedList<Node<T>> neighbors;

        public Node(T data) {
            this.data = data;
        }

        public void addNeighbor(Node<T> n) {
            neighbors.addAtTail(n);
        }
    }

    public AdjacencyListUndirectedGraph() {
        nodes = new SinglyLinkedList<Node<T>>();
    }

    public void addNode(T data) {
        if (hasNode(data)) {
            // do nothing
        } else {
            Node<T> node = new Node<T>(data);
            nodes.addAtTail(node);
        }
    }

    public void addEdge(T d1, T d2) throws MissingNodeException {
        SinglyLinkedList<Node<T>> neighbors1 = neighborsOf(d1);
        if (neighbors1 == null) throw new MissingNodeException(
        SinglyLinkedList<Node<T>> neighbors2 = neighborsOf(d2);
    }

    public boolean hasNode(T data) {  // O(n)
        Node<T> node = nodeOf(data);
        if (node == null) {
            return false;
        } else {
            return true;
        }
    }

    public boolean hasEdge(T d1, T d2) {  // TODO figure out runtime of methods in this class, probably a lot of unnecessary iteration
        // if (!hasNode(d1)) return false;  // this might be unnecessary, we can just get the neighbors and check for null
        SinglyLinkedList<Node<T>> neighbors1 = neighborsOf(d1);
        if (neighbors1 == null) {
            return false;  // d1 doesn't exist in the graph
        }
        return neighbors1.contains(nodeOf(d2));
    }

    public SinglyLinkedList<Node<T>> neighborsOf(T data) {
        Node<T> node = nodeOf(data);
        if (node != null) {
            return node.neighbors;
        } else {
            return null;
        }
    }

    private Node<T> nodeOf(T data) {
        //Node<T> node = nodes.head;
        //int size = nodes.size();
        //for (int i = 0; i < size; i++) {
        nodes.resetIterator();
        while (nodes.hasNext()) {
            Node<T> thisNode = nodes.getNext();
            if (thisNode.data == data) return thisNode;
        }
        return null;
    }

    public String toString() {
        return "TODO";
    }

    public static void main(String[] args) {
        AdjacencyListUndirectedGraph<Integer> g = new AdjacencyListUndirectedGraph<Integer>();
        for (int i = 4; i < 10; i++) {
            int j = (i*i) % 10;
            if (!g.hasNode(i)) g.addNode(i);
            if (!g.hasNode(j)) g.addNode(j);
            if (!g.hasEdge(i, j)) g.addEdge(i, j);
            System.out.printf("graph is now:\n%s\n", g.toString());
        }
    }
}
