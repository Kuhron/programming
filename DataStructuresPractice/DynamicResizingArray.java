public class DynamicResizingArray<T> {
    private T[] array = (T[]) new Object[1];
    private int capacity = 1;
    public int length = 0;

    public DynamicResizingArray() {
        // nothing
    }

    public void insert(T item) {
        array[length] = item;
        length++;
        resize();
    }

    private void resize() {
        if (length == capacity) {
            // resize to double
            int newCapacity = 2 * capacity;
            T[] newArray = (T[]) new Object[newCapacity];
            copyItems(array, newArray);
            array = newArray;
        }
    }

    private void copyItems(T[] array, T[] newArray) {
        for (int i = 0; i < array.length; i++) {
            newArray[i] = array[i];
        }
    }

    public void push(T item) {
        int index = length;
        setItem(index, item);
    }

    public void setItem(int index, T item) {
        array[index] = item;
    }
}
