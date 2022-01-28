public class SinglyLinkedList<T> {
    private Node<T> head;
    private Node<T> tail;
    private int size;
    private int iterationIndex;
    private Node<T> iterationNode;

    private class Node<T> {
        T data;
        Node<T> next;

        public Node(T data, Node<T> next) { // don't need T in the constructor declaration, it will be inferred from the params
            this.data = data;  // need this.varname to get past the fact that "data" as an instance variable is shadowed by "data" as a param in this scope; if you made the param with a different name e.g. _data, you could just do: data = _data;
            this.next = next;
        }
    }

    public SinglyLinkedList() {  // again don't need T in the contsructor declaration
        head = null;
        tail = null;
        size = 0;

        iterationIndex = 0;
        iterationNode = null;
    }

    public int size() {
        return size;
    }

    public void addAtHead(T data) {  // O(1)
        Node<T> newNext = head;
        Node<T> node = new Node<T>(data, newNext);
        head = node;
        if (size == 0) {
            tail = node;
        }
        size++;
    }

    public T removeAtHead() throws EmptyListException {  // O(1), just move a pointer
        if (size == 0) {
            throw new EmptyListException("list is empty");
        }

        Node<T> newHead = head.next;
        T data = head.data;
        head = newHead;
        size--;

        return data;
    }

    public void addAtTail(T data) {  // O(1), just move a pointer
        Node<T> newNode = new Node<T>(data, null);

        if (size == 0) {
            // make head and tail both the new node
            head = newNode;
            tail = newNode;
        } else {
            tail.next = newNode;
            tail = newNode;
        }
        size++;
    }

    public T removeAtTail() {  // O(n)
        Node<T> newTail = nodeAt(size-2);  // this is why removeAtTail is O(n) for singly linked list, since we don't have tail.prev
        T data = tail.data;
        newTail.next = null;
        tail = newTail;
        size--;
        return data;
    }

    private Node<T> nodeAt(int index) {  // O(n)
        Node<T> node = head;
        for (int i = 0; i < index; i++) {
            node = node.next;
        }
        return node;
    }

    public boolean contains(T data) {
        Node<T> node = head;
        for (int i = 0; i < size; i++) {
            if (node.data == data) return true;
            node = node.next;
        }
        return false;
    }

    public int indexOf(T data) {
        Node<T> node = head;
        for (int i = 0; i < size; i++) {
            if (node.data == data) return i;
            node = node.next;
        }
        return -1;
    }

    public boolean hasNext() {
        return iterationIndex < size - 1;
    }

    public T getNext() {
        T data = iterationNode.data;
        iterationIndex++;
        iterationNode = iterationNode.next;
        return data;
    }

    public void resetIterator() {
        iterationIndex = 0;
        iterationNode = head;
    }

    public String toString() {  // O(n)
        String s = "[";
        Node<T> node = head;
        for (int i = 0; i < size; i++) {
            T data = node.data;
            String newS = data.toString();
            if (i < size - 1) {
                newS += ", ";
            }
            s += newS;
            node = node.next;
        }
        s += "]";
        return s;
    }

    public static void main(String[] args) throws EmptyListException {
        SinglyLinkedList<Integer> l = new SinglyLinkedList<Integer>();
        // apparently you can't do generics of primitive types, which is annoying
        for (Integer i = 0; i < 5; i++) {
            l.addAtHead(i);
            System.out.printf("added %d\n", i);
            System.out.printf("size is now %d\n", l.size());
        }
        System.out.println(l.toString());
        System.out.printf("size: %d\n", l.size());

        for (int i = 0; i < 3; i++) {
            Integer x = l.removeAtHead();
            System.out.printf("removed %d\n", x);
            System.out.printf("size is now %d\n", l.size());
        }
        System.out.println(l.toString());
        System.out.printf("size: %d\n", l.size());

        for (Integer i = 17; i < 22; i++) {
            l.addAtTail(i);
            System.out.printf("added %d\n", i);
            System.out.printf("size is now %d\n", l.size());
        }
        System.out.println(l.toString());
        System.out.printf("size: %d\n", l.size());

        for (int i = 0; i < l.size(); i++) {
            Integer x = l.removeAtTail();
            System.out.printf("removed %d\n", x);
            System.out.printf("size is now %d\n", l.size());
        }
        System.out.println(l.toString());
        System.out.printf("size: %d\n", l.size());
    }
}


