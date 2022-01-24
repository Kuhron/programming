public class Stack<T> {
    private int size;
    private SinglyLinkedList<T> list;

    public Stack() {
        size = 0;
        list = new SinglyLinkedList<T>();
    }

    public void push(T data) {
        list.addAtHead(data);
        size++;
    }

    public T pop() throws EmptyListException {
        T data = list.removeAtHead();
        size--;
        return data;
    }

    public int size() {
        System.out.println("called size");
        return size;
    }

    public String toString() {
        return list.toString();
    }

    public static void main(String[] args) throws EmptyListException {
        Stack<Integer> s = new Stack<Integer>();
        for (int i = 2; i < 7; i++) {
            s.push(i);
            System.out.printf("stack is: %s with size %d\n", s.toString(), s.size());
        }
        // for (int i = 0; i < s.size(); i++) { // apparently it calls s.size() every time to check the condition, rather than evaluating s.size() at the beginning and using that int the whole time
        int size = s.size();
        for (int i = 0; i < size; i++) {
            Integer x = s.pop();
            System.out.printf("popped %d off the stack\n", x);
            System.out.printf("stack is: %s with size %d\n", s.toString(), s.size());
        }
    }
}
