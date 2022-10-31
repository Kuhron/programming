public class BinarySearchTreeNode<T extends Comparable<T>> {
    public Integer key;
    public T value;
    public BinarySearchTreeNode<T> parent;
    public BinarySearchTreeNode<T> left;
    public BinarySearchTreeNode<T> right;

    public BinarySearchTreeNode(Integer key, T value) {
        this.key = key;
        this.value = value;
    }

    public void insertChild(Integer key, T value) {
        if (key <= this.key) {
            // it goes on the left
            if (left == null) {
                // put it as left child
                left = new BinarySearchTreeNode<>(key, value);
            } else {
                // put it as a child of left child
                left.insertChild(key, value);
            }
        } else {
            // it goes on the right
            if (right == null) {
                // put it as right child
                right = new BinarySearchTreeNode<>(key, value);
            } else {
                // put it as a child of right child
                right.insertChild(key, value);
            }
        }
    }

    public BinarySearchTreeNode<T> getSuccessor() {
        if (right == null) {
            return parent;
        } else {
            BinarySearchTreeNode<T> current = right;
            while (current.left != null) {
                current = current.left;
            }
            return current;
        }
    }

    public void printInOrder() {
        if (left != null) {
            left.printInOrder();
        }
        printValue();
        if (right != null) {
            right.printInOrder();
        }
    }

    public void printPreOrder() {
        printValue();
        if (left != null) {
            left.printPreOrder();
        }
        if (right != null) {
            right.printPreOrder();
        }
    }

    public void printPostOrder() {
        if (left != null) {
            left.printPostOrder();
        }
        if (right != null) {
            right.printPostOrder();
        }
        printValue();
    }

    public void printKeyAndValue() {
        System.out.printf("%d:%s ", key, value);
    }

    public void printValue() {
        System.out.printf("%s ", value);
    }
}

