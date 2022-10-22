import java.util.Arrays;


public class BinaryHeap<T extends Comparable<T>> {
    private DynamicResizingArray<T> array;

    public BinaryHeap() {
        array = new DynamicResizingArray<T>();
    }

    public BinaryHeap(T[] items) {
        array = new DynamicResizingArray<T>();
        for (int i = 0; i < items.length; i++) {
            T item = items[i];
            // System.out.printf("inserting %d\n", item);
            this.insert(item);
            System.out.println("array is now:");
            print();
        }
    }

    public int getLength() {
        return array.length;  // NOT array.array.length, which is the capacity (also array.capacity)
    }

    public T extractMax() {
        // extract root, move last item to root position
        // then rebalance by swapping item with largest child until heap property is satisfied
        T item = array.extractFirstAndMoveLast();
        heapifyDown(0);
        return item;
    }

    public void insert(T item) {
        // put it at the end and then swap up with parent as much as needed for heap property
        array.insert(item);
        heapifyUp(array.length - 1);
    }

    private void heapifyUp(int index) {
        // starting with item at index, swap up until heap is good
        if (index == 0) {
            // nothing to do
            return;
        }

        int parentIndex = getParentIndex(index);
        T parent = array.getItem(parentIndex);
        T item = array.getItem(index);

        if (parent == null) {
            throw new RuntimeException("parent is null");
        }
        if (item == null) {
            throw new RuntimeException("item is null");
        }
        // System.out.printf("parent: %d; item: %d\n", parent, item);
        boolean needToSwap = parent.compareTo(item) < 0;
        if (needToSwap) {
            System.out.printf("heapifyUp swapping %d and %d\n", parent, item);
            array.swap(index, parentIndex);
            heapifyUp(parentIndex);
        }
    }

    private void heapifyDown(int index) {
        // starting with item at index, swap down until heap is good
        int leftChildIndex = getLeftChildIndex(index);
        int rightChildIndex = getRightChildIndex(index);
        T left = array.getItem(leftChildIndex);
        T right = array.getItem(rightChildIndex);

        T bigger;
        int biggerIndex;
        if (left == null) {
            if (right != null) {
                throw new RuntimeException("left is null but right is not");
            }
            // don't do any swaps
            return;
        } else if (right == null) {
            // the largest child is the only child, left
            bigger = left;
            biggerIndex = leftChildIndex;
        } else {
            boolean leftGeqRight = left.compareTo(right) >= 0;
            bigger = leftGeqRight ? left : right;
            biggerIndex = leftGeqRight ? leftChildIndex : rightChildIndex;
        }

        T item = array.getItem(index);
        boolean needToSwap = item.compareTo(bigger) < 0;
        if (needToSwap) {
            System.out.printf("heapifyDown swapping %d and %d\n", item, bigger);
            array.swap(index, biggerIndex);
            heapifyDown(biggerIndex);
        }
    }

    private int getParentIndex(int index) {
        return Math.floorDiv(index - 1, 2);
    }

    private int getLeftChildIndex(int index) {
        return index * 2 + 1;
    }

    private int getRightChildIndex(int index) {
        return index * 2 + 2;
    }

    public void print() {
        array.print();
    }

    public static <T2 extends Comparable<T2>> Comparable[] heapSort(T2[] items) {
        BinaryHeap<T2> heap = new BinaryHeap<>(items);
        T2[] sorted = (T2[]) new Comparable[items.length];
        for (int i = 0; i < items.length; i++) {
            T2 next = heap.extractMax();
            sorted[i] = next;
        }
        return sorted;
    }

    public static Integer[] copyComparableArrayToIntegerArray(Comparable[] a) {
        // stupid Java crap
        Integer[] b = new Integer[a.length];
        for (int i = 0; i < a.length; i++) {
            b[i] = (Integer) a[i];
        }
        return b;
    }

    public static void main(String[] args) {
        Integer[] a = {30, 41, 5, 2, 27, 11, 70, 35, 56, 8, 14, 33, 74, 94};
        // BinaryHeap<Integer> heap = new BinaryHeap<>(a);
        // heap.print();
        Comparable[] rawSorted = heapSort(a);
        Integer[] sorted = copyComparableArrayToIntegerArray(rawSorted);
        System.out.println(Arrays.toString(sorted));
    }
}
