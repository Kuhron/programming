public class DynamicResizingArray<T> {
    private T[] array = (T[]) new Object[1];
    private int capacity = 1;
    public int length = 0;

    public DynamicResizingArray() {
        // nothing
    }

    public void insert(T item) {
        System.out.printf("inserting item %d at index %d\n", item, length);
        setItem(length, item);
        length++;
        resize();
    }

    private void resize() {
        int newCapacity;
        if (length == capacity) {
            // resize to double
            System.out.println("doubling array");
            newCapacity = 2 * capacity;
        } else if (length * 2 == capacity) {
            // resize to half
            System.out.println("halving array");
            newCapacity = length;
        } else if (length > capacity) {
            throw new RuntimeException("shouldn't happen");
        } else {
            // no need to resize right now
            return;
        }

        T[] newArray = (T[]) new Object[newCapacity];
        copyItems(array, newArray);
        array = newArray;
        capacity = newCapacity;
    }

    private void copyItems(T[] oldArray, T[] newArray) {
        int newLength = Math.min(oldArray.length, newArray.length);
        for (int i = 0; i < newLength; i++) {
            newArray[i] = oldArray[i];
        }
    }

    public T getItem(int index) {
        if (0 <= index && index < length) {
            return array[index];
        }
        return null;
    }

    public void setItem(int index, T item) {
        array[index] = item;
    }

    public T extractFirstAndMoveLast() {
        T item = array[0];
        array[0] = array[length - 1];
        array[length - 1] = null;
        length--;
        resize();
        return item;
    }

    public void swap(int index1, int index2) {
        T x1 = array[index1];
        T x2 = array[index2];
        setItem(index1, x2);
        setItem(index2, x1);
    }

    public void print() {
        String s = "[";
        for (int i = 0; i < length; i++) {
            s += array[i].toString();
            if (i < length - 1) {
                s += ", ";
            }
        }
        s += "]";
        if (capacity > length) {
            String nNulls = Integer.toString(capacity - length);
            s += " (plus " + nNulls + " empty slots)";
        }
        System.out.println(s);
    }
}
