// can do this with a singly linked list without incurring O(n) removal time if you put them on at the tail and remove them at the head (since removeAtTail() is the bad one for singly-LLs, since it has to go traverse from the beginning all over again to find what the new tail is going to be)

public class Queue<T> {
    private int size;
    private SinglyLinkedList<T> list;

    public Queue() {
        size = 0;
        list = new SinglyLinkedList<T>();
    }

    public void enqueue(T data) {
        // add the data at the tail of the list and remove at the head, so they are both O(1)
        list.addAtTail(data);
        size++;
    }

    public T dequeue() throws EmptyListException {
        // add the data at the tail of the list and remove at the head, so they are both O(1)
        T data = list.removeAtHead();
        size--;
        return data;
    }

    public int size() {
        return size;  // this doesn't shadow the function name somehow
    }

    public String toString() {  // O(n)
        return list.toString();
    }

    public static void main(String[] args) throws EmptyListException {
        Queue<Integer> q = new Queue<Integer>();
        for (int i = 5; i < 13; i++) {
            Integer x = (Integer) (int) Math.floor(Math.pow(i, 2));  // stupid casting, why can't I just do straight to Integer from double output
            q.enqueue(x);
            System.out.printf("enqueued %d, queue is now %s\n", x, q.toString());
        }
        int size = q.size();
        for (int i = 0; i < size; i++) {
        // for (int i = 0; i < q.size(); i++) {  // example of calling size() every loop and so i reaches it too quickly because the i < q.size() is true before the original size is reached
            Integer x = q.dequeue();
            System.out.printf("dequeued %d, queue is now %s\n", x, q.toString());
        }
    }
}
