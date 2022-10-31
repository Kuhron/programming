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

    public void insert(Integer key, T value) {
        if (root == null) {
            // set item as root and that's all we have to worry about
            root = new BinarySearchTreeNode<>(key, value);
        } else {
            root.insertChild(key, value);
        }
    }

    public void delete(T item) {
        // TODO
    }

    public T getSuccessorOfItem(T item) {
        BinarySearchTreeNode<T> node = findNode(item);
        BinarySearchTreeNode<T> successorNode = node.getSuccessor();
        if (successorNode == null) {
            return null;
        } else {
            return successorNode.value;
        }
    }

    public BinarySearchTreeNode<T> findNode(T item) {
        BinarySearchTreeNode<T> current = root;
        while (current.value != item ?
    }

    public T find(T item) {
        BinarySearchTreeNode<T> node = findNode(item);
        if (node == null) {
            return null;
        } else {
            return node.value;
        }
    }

    public void printInOrder() {
        root.printInOrder();
        System.out.print("\n");
    }

    public void printPreOrder() {
        root.printPreOrder();
        System.out.print("\n");
    }

    public void printPostOrder() {
        root.printPostOrder();
        System.out.print("\n");
    }

    public static void main(String[] args) {
        // BinarySearchTree<Integer> = new BinarySearchTree<>();
        Integer[] nums = {30, 4, 15, 7, 70, 21, 27, 48, 13, 8, 37, 56, 41, 19, 20};
        BinarySearchTree<Integer> tree = new BinarySearchTree<>(nums, nums);
        tree.printInOrder();
        Integer found27 = tree.find(27);
        Integer found51 = tree.find(51);
        System.out.printf("27 in tree: %d; 51 in tree: %d\n", found27, found51);

        tree.delete(48);
        tree.printInOrder();
    }
}
