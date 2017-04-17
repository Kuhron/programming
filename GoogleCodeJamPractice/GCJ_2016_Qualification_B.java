import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


class GCJ_2016_Qualification_B {
    static boolean[][] getStackArrays(Scanner scanner) {
        List<String> lines = new ArrayList<>();
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            lines.add(line);
        }
        boolean[][] result = new boolean[lines.size() - 1][];  // leave sizes of second elements unspecified, and allocate them later
        String acceptableChars = "+-";
        for (int i = 0; i < lines.size() - 1; i++) {
            String line = lines.get(i + 1);
            boolean[] bools = new boolean[line.length()];
            for (int j = 0; j < line.length(); j++) {
                char c = line.charAt(j);
                boolean isAcceptable = acceptableChars.indexOf(c) >= 0;
                if (!isAcceptable) {
                    throw new RuntimeException("strs must only contain '+' and '-'");
                }
                bools[j] = (c == '+');
            }
            result[i] = bools;
        }
        return result;
    }

    static int getNumFlips(boolean[] stack) {
        if (allPluses(stack)) {
            return 0;
        }
        boolean[] strippedStack = getStrippedStack(stack);
        return 1 + getNumSignChanges(strippedStack);
    }

    static boolean allPluses(boolean[] stack) {
        for (boolean x : stack) {
            if (!x) {
                return false;
            }
        }
        return true;
    }

    static boolean[] getStrippedStack(boolean[] stack) {
        // shouldn't be all true
        int endIndex = stack.length;
        while (stack[endIndex - 1]) {
            endIndex--;
            if (endIndex == 0) {
                throw new RuntimeException("tried to strip all-plus stack");
            }
        }

        // at end, no need to flip back since we will just count sign changes!
        return Arrays.copyOfRange(stack, 0, endIndex);
    }

    static int getNumSignChanges(boolean[] stack) {
        if (stack.length <= 1) {
            return 0;
        }
        boolean currentSign = stack[0];
        int count = 0;
        for (int i = 1; i < stack.length; i++) {
            if (stack[i] != currentSign) count++;
            currentSign = stack[i];
        }
        return count;
    }

    static void writeOutput(int[] output) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter(new File("output.txt"));
        for (int i = 0; i < output.length; i++) {
            int x = output[i];
            writer.println(String.format("Case #%d: %s", i + 1, x));
        }
        writer.flush();
    }

    public static void main(String[] args) throws FileNotFoundException {
        String inputFilepath = args[0];
        Scanner scanner = new Scanner(new File(inputFilepath));
        boolean[][] stackArrays = getStackArrays(scanner);
        // System.out.println(Arrays.deepToString(stackArrays));
        int[] output = new int[stackArrays.length];
        for (int i = 0; i < stackArrays.length; i++) {
            boolean[] stack = stackArrays[i];
            int numFlips = getNumFlips(stack);
            output[i] = numFlips;
            // System.out.println(numFlips);
        }
        writeOutput(output);
    }
}