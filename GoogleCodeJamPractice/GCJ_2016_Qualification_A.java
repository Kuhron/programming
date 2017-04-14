import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.lang.Integer;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;


class GCJ_2016_Qualification_A {
    static int[] getNumbers(Scanner scanner) {
        List<Integer> result = new ArrayList<>();
        while (scanner.hasNextInt()) {
            result.add(scanner.nextInt());
        }
        // Integer[] temp = result.toArray(new Integer[result.size()]);
        int[] arrayResult = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            arrayResult[i] = result.get(i);
        }
        return arrayResult;
    }

    static String getOutputForNumber(int n) {
        if (insomniaOccurs(n)) {
            return "INSOMNIA";
        }
        
        boolean[] digitsSeen = new boolean[10];
        int lastNum = 0;
        while (!allTrue(digitsSeen)) {
            lastNum += n;
            digitsSeen = copyNewDigits(lastNum, digitsSeen);
            // System.out.println(String.format("next number %d gives array %s", lastNum, Arrays.toString(digitsSeen)));
        }

        return Integer.toString(lastNum);
    }

    static boolean insomniaOccurs(int n) {
        // when does insomnia occur?
        // will get all digits in ones place if number is coprime with 10, i.e., ending in 1, 3, 7, 9

        return (n == 0);  // TODO
    }

    static boolean allTrue(boolean[] array) {
        for (boolean x : array) {
            if (!x) {
                return false;
            }
        }
        return true;
    }

    static boolean[] copyNewDigits(int newNum, boolean[] digitsSeen) {
        String s = Integer.toString(newNum);
        for (int i = 0; i < s.length(); i++) {
            int digit = Character.getNumericValue(s.charAt(i));
            digitsSeen[digit] = true;
        }
        return digitsSeen;
    }

    public static void validateOutput(String[] output) throws FileNotFoundException {
        File expectedOutputFile = new File("expected.txt");
        Scanner scanner = new Scanner(expectedOutputFile);
        int lineIndex = 0;
        while (scanner.hasNextLine()) {
            String expectedLine = scanner.nextLine();
            String gotLine = output[lineIndex];
            if (!expectedLine.equals(gotLine)) {
                throw new RuntimeException(String.format("incorrect output! expected \"%s\"; got \"%s\"", expectedLine, gotLine));
            }
            lineIndex++;
        }
    }

    public static void writeOutput(String[] output) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter(new File("output.txt"));
        for (String s : output) {
            writer.println(s);
        }
        writer.flush();
    }

    public static void main(String[] args) throws FileNotFoundException {
        String inputFilePath = args[0];
        File inputFile = new File(inputFilePath);
        Scanner scanner = new Scanner(inputFile);
        int[] numbers = getNumbers(scanner);
        int nCases = numbers[0];
        String[] output = new String[nCases];
        for (int i = 1; i < numbers.length; i++) {
            String newOutput = getOutputForNumber(numbers[i]);
            output[i - 1] = String.format("Case #%d: %s", i, newOutput);
        }

        // validateOutput(output);
        writeOutput(output);
    }
}