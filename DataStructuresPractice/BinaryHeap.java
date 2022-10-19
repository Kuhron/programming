public class BinaryHeap<T> {
    private DynamicResizingArray<T> array = new DynamicResizingArray<T>();

    public BinaryHeap() {
        // nothing
    }

    public T extractMax() {
        return null; // TODO
    }

    public void insert(T item) {
        // put it at the end and then swap up with parent as much as needed for heap property
        // TODO
    }

    public static void main(String[] args) {
        BinaryHeap heap = new BinaryHeap();
    }
}
