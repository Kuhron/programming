import java.lang.reflect.Field;
import java.util.Random;
import java.io.*;

public class JavaIntegerCache {
    public static void main(String[] args) throws Exception {
        Class cache = Integer.class.getDeclaredClasses()[0];
        Field c = cache.getDeclaredField("cache");
        c.setAccessible(true);
        Integer[] array = (Integer[]) c.get(cache);
        // array = rearrangeArray(array);
        // array[132] = array[133];
        rearrangeArray(array, 5);  // does it pass by ref or value?  // looks like ref (same object) if you mutate in place

        String outputFp = "JavaIntegerCacheOutput.txt";
        try {
            File outputFile = new File(outputFp);
            if (outputFile.createNewFile()) {
                System.out.println("File created: " + outputFile.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        try {
            FileWriter myWriter = new FileWriter(outputFp);
            Integer[][] additions = {
                {2, 2}, {1, 4}, {0, 0}, {-1, 1}, {15, 30}
            };
            for (Integer[] pair : additions) {
                Integer a = pair[0];
                Integer b = pair[1];
                myWriter.write(a + " + " + b + " = " + (a+b) + "\n");
            }
            for (int i = 0; i < array.length; i++) {
                myWriter.write("array[" + i + "] = " + array[i] + "\n");
            }


            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static void rearrangeArray(Integer[] array, int nTimes) {
        Random random = new Random();  // this is so Java
        // Integer[] newArray = new Integer[array.length];
        for (int stepI = 0; stepI < nTimes; stepI++) {
            for (int i = 0; i < array.length; i++) {
                // don't care about preventing repetitions right now
                int arrayIndexToAccess = random.nextInt(array.length);
                // newArray[i] = array[arrayIndexToAccess];
                array[i] = array[arrayIndexToAccess];  // try modifying in place so cache is actually changed
            }
        }
        // return newArray;
    }
}
