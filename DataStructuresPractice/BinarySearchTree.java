import java.util.Arrays;



public class BinarySearchTree<T extends Comparable<T>> {
    public BinarySearchTreeNode<T> root;

    public BinarySearchTree() {
        root = null;
    }

    public BinarySearchTree(Integer[] keys, T[] items) {
        root = null;
        if (keys.length != items.length) {
            throw new RuntimeException("keys and items must be same length");
        }
        for (int i = 0; i < keys.length; i++) {
            Integer key = keys[i];
            T value = items[i];
            insert(key, value);
        }
    }

    public BinarySearchTree(Integer[] items) {
        this(items, items);  // call another constructor
    }

    public void insert(Integer key, T value) {
        if (root == null) {
            // set item as root and that's all we have to worry about
            root = new BinarySearchTreeNode<>(key, value);
        } else {
            root.insertChild(key, value);
        }
    }

    public void delete(T item) {
    }

    public T getSuccessorOfItem(T item) {
        // might need iterator sentinel thing to keep track of where we are in the tree so we don't have to re-find everything
        return null;
    }

    public T find(T item) {
        return null;
    }

    public void printInOrder() {
        
    }

    public void printPreOrder() {
    }

    public void printPostOrder() {
        
    }

    public static void main(String[] args) {
        // BinarySearchTree<Integer> = new BinarySearchTree<>();
        Integer[] nums = {30, 4, 15, 7, 70, 21, 27, 48, 13, 8, 37, 56, 41, 19, 20};
        BinarySearchTree<Integer> tree = new BinarySearchTree<>(nums);
    }
}
