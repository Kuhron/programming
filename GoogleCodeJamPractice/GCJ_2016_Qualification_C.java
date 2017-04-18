import java.lang.Math;
import java.util.ArrayList; 
import java.util.Arrays;
import java.util.List;
import java.util.Random;


class PrimeNumbers {
    private static List<Integer> knownPrimes = new ArrayList<>(Arrays.asList(2, 3));
    private static Integer biggestKnownPrime = 3;

    public PrimeNumbers() {}

    public int getLowestPrimeFactor(int n) {
        // don't calculate more primes unnecessarily; if can find a divisor in the known list, return it
        // but if finish known list and haven't reached sqrt(n), iterate over next odd numbers,
            // first checking if they're prime (if they are, add to list and then check if they divide n)

        for (Integer p : this.knownPrimes) {
            if (n % p == 0) {
                return p;
            }
        }

        int nextPrimeCandidate = this.biggestKnownPrime;
        while (Math.sqrt(n) > this.biggestKnownPrime) {
            nextPrimeCandidate += 2;
            if (getLowestPrimeFactor(nextPrimeCandidate) == nextPrimeCandidate) {
                this.knownPrimes.add(nextPrimeCandidate);
                this.biggestKnownPrime = nextPrimeCandidate;
                if (n % nextPrimeCandidate == 0) {
                    return nextPrimeCandidate;
                }
            }
        }

        // although n is prime here, neglect to add it to the list because we would be skipping primes in between
        return n;
    }
}

class GCJ_2016_Qualification_C {
    static int convertToBase(boolean[] coin, int base) {
        int result = 0;
        for (int i = 0; i < coin.length; i++) {
            boolean bit = coin[coin.length - i - 1];
            int power = i;
            result += bit ? (Math.pow(base, power)) : 0;
        }
        return result;
    }

    static boolean[] convertStringToCoin(String s) {
        boolean[] result = new boolean[s.length()];
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c != '1' && c != '0') {
                throw new RuntimeException(String.format("invalid string %s (char %c at index %d)", s, c, i));
            }
            result[i] = (c == '1');
        }
        return result;
    }

    static String convertCoinToString(boolean[] coin) {
        String result = "";
        for (int i = 0; i < coin.length; i++) {
            result += (coin[i] ? "1" : "0");
        }
        return result;
    }

    static int[] getBaseValues(boolean[] coin) {
        int[] result = new int[9];
        for (int i = 2; i < 11; i++) {
            int n = convertToBase(coin, i);
            result[i - 2] = n;
        }
        return result;
    }

    static int[] getNonTrivialDivisors(int[] coinValues, PrimeNumbers primes) {
        int[] result = new int[9];
        for (int i = 0; i < 9; i++) {
            int val = coinValues[i];
            int divisor = primes.getLowestPrimeFactor(val);
            if (divisor == val) {
                // not a jamcoin because value in one of the bases is prime
                return null;
            }
            result[i] = divisor;
        }
        return result;
    }

    static boolean[] getRandomCoin(int length) {
        boolean[] result = new boolean[length];
        result[0] = result[length - 1] = true;
        Random random = new Random();
        for (int i = 1; i < length - 1; i++) {
            result[i] = random.nextBoolean();
        }
        return result;
    }

    static String[] getOutput(int coinLength, int nCoins, PrimeNumbers primes) {
        String[] result = new String[nCoins];
        List<String> seenCoins = new ArrayList<>();
        // for (int i = 0; i < nCoins; i++) {
            boolean[] coin;
            int[] values;
            int[] divisors;
            int i = 0;
            while (true) {
                coin = getRandomCoin(coinLength);
                values = null;
                divisors = null;  // was not being overwritten for some reason
                String s = convertCoinToString(coin);
                boolean coinSeen = seenCoins.contains(s);
                if (coinSeen) continue;
                seenCoins.add(s);
                values = getBaseValues(coin);
                divisors = getNonTrivialDivisors(values, primes);
                if (divisors == null) continue;
                for (int divisor : divisors) {
                    s += String.format(" %d", divisor);
                }
                
                System.out.println(i + ". found new coin: " + s);
                result[i] = s;
                i++;
                if (i >= nCoins) break;
            }
        // }
        return result;
    }

    public static void main(String[] args) {
        PrimeNumbers primes = new PrimeNumbers();
        int coinLength = Integer.parseInt(args[0]);
        int nCoins = Integer.parseInt(args[1]);
        String[] output = getOutput(coinLength, nCoins, primes);
    }
}