public class BinarySearchTreeNode<T extends Comparable<T>> {
    public Integer key;
    public T value;
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
}


