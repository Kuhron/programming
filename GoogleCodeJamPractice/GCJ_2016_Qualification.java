import java.io.File;
import java.io.FileNotFoundException;
import java.lang.Integer;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;


class GCJ_2016_Qualification {
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
        int lastNum = n;
        while (!allTrue(digitsSeen)) {
            lastNum += n;
            digitsSeen = copyNewDigits(lastNum, digitsSeen);
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

    public static void main(String[] args) throws FileNotFoundException {
        File file = new File("GCJ_2016_Qualification_input.txt");
        Scanner scanner = new Scanner(file);
        int[] numbers = getNumbers(scanner);
        int nCases = numbers[0];
        String[] output = new String[nCases];
        for (int i = 1; i < numbers.length; i++) {
            System.out.println(numbers[i]);
            String newOutput = getOutputForNumber(numbers[i]);
            output[i - 1] = String.format("Case #%d: %s", i, newOutput);
        }

        File expectedOutputFile = new File("GCJ_2016_Qualification_expected.txt");
        scanner = new Scanner(expectedOutputFile);
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
}