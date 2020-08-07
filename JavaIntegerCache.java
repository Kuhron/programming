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
        rearrangeArray(array, 1);  // does it pass by ref or value?  // looks like ref (same object) if you mutate in place

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
            // ints above 127 are not in the cache array
            myWriter.write("array[128] = " + array[128] + " (should be 0)\n");  // note that referencing the int may get the array item instead
            myWriter.write("0 direct reference = " + 0 + "\n");
            myWriter.write("128 direct reference = " + 128 + "\n");

            myWriter.write("array[129] = " + array[129] + " (should be 1)\n");
            myWriter.write("1 direct reference = " + 1 + "\n");
            myWriter.write("129 direct reference = " + 129 + "\n");

            Integer[][] additions = {
                {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},
                {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4},
                {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4},
                {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4},
                {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4},
            };
            myWriter.write("0 through 4 addition table:\n");
            for (Integer[] pair : additions) {
                Integer a = pair[0];
                Integer b = pair[1];
                myWriter.write(a + " + " + b + " = " + (a+b) + "\n");
            }

            myWriter.write("loop from Integer 0 <= i < 5:\n");
            // for (Integer i = 0; i < (Integer)(5); i++) { // DON'T DO THIS! can cause infinite loop, couldn't kill with -15 or -11, had to kill -9
            Integer toPrintIntegerPlusPlus = 0;  // increment the Integer, but loop var is int, so it will run 5 times and not do god knows what
            Integer toPrintIntegerPlusEqualsOne = 0;
            for (int i = 0; i < 5; i++) {
                myWriter.write("i=" + i + "; IntegerPlusPlus=" + toPrintIntegerPlusPlus + "; IntegerPlusEqualsOne=" + toPrintIntegerPlusEqualsOne + "\n");
                toPrintIntegerPlusPlus++;
                toPrintIntegerPlusEqualsOne += 1;
            }

            int fourInt = 4;
            Integer fourInteger = 4;
            myWriter.write("four as int = " + fourInt + "\n");
            myWriter.write("four as Integer = " + fourInteger + "\n");
            myWriter.write("adding the fours = " + (fourInt + fourInteger) + "\n");

            fourInt++;
            fourInteger++;
            myWriter.write("fourInt++ -> " + fourInt + "\n");
            myWriter.write("fourInteger++ -> " + fourInt + "\n");

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
