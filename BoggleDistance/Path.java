public class Path<T> {
    // basically just an ArrayList of T objects, but construed as a path through the Boggle grid of such objects (in the Boggle game the objects are all char)
    private ArrayList<T> list;

    public Path<T>(ArrayList<T> list) {
        this.list = list;
    }

    public Path<T>(T[] array) {
        this.list = new ArrayList<T>(Arrays.asList(array));
    }
}
