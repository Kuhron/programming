import java.lang.Math;
import java.math.BigInteger;
import java.util.ArrayList; 
import java.util.Arrays;
import java.util.List;
import java.util.Random;


class PrimeNumbers {
    private static List<BigInteger> knownPrimes = new ArrayList<BigInteger>(Arrays.asList(new BigInteger("2"), new BigInteger("3")));
    private static BigInteger biggestKnownPrime = new BigInteger("3");

    public PrimeNumbers() {}

    public BigInteger getLowestPrimeFactor(BigInteger n) {
        // don't calculate more primes unnecessarily; if can find a divisor in the known list, return it
        // but if finish known list and haven't reached sqrt(n), iterate over next odd numbers,
            // first checking if they're prime (if they are, add to list and then check if they divide n)

        for (BigInteger p : this.knownPrimes) {
            if (n.mod(p).equals(BigInteger.ZERO)) {
                return p;
            }
        }

        BigInteger nextPrimeCandidate = this.biggestKnownPrime;
        while (BigIntMath.getCeilingSqrt(n) > this.biggestKnownPrime) {
            nextPrimeCandidate = nextPrimeCandidate.add(new BigInteger("2"));
            if (getLowestPrimeFactor(nextPrimeCandidate).equals(nextPrimeCandidate)) {
                this.knownPrimes.add(nextPrimeCandidate);
                this.biggestKnownPrime = nextPrimeCandidate;
                if (n.mod(nextPrimeCandidate).equals(BigInteger.ZERO)) {
                    return nextPrimeCandidate;
                }
            }
        }

        // although n is prime here, neglect to add it to the list because we would be skipping primes in between
        return n;
    }
}

class BigIntMath {
    public static BigInteger getCeilingSqrt(BigInteger x) {
        ;
    }
}

class GCJ_2016_Qualification_C {
    static BigInteger convertToBase(boolean[] coin, int base) {
        BigInteger result = BigInteger.ZERO;
        for (int i = 0; i < coin.length; i++) {
            boolean bit = coin[coin.length - i - 1];
            int power = i;
            BigInteger addend = bit ? BigInteger.valueOf((int) Math.pow(base, power)) : BigInteger.ZERO;
            result = result.add(addend);
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

    static BigInteger[] getBaseValues(boolean[] coin) {
        BigInteger[] result = new BigInteger[9];
        for (int i = 2; i < 11; i++) {
            BigInteger n = convertToBase(coin, i);
            result[i - 2] = n;
        }
        return result;
    }

    static BigInteger[] getNonTrivialDivisors(BigInteger[] coinValues, PrimeNumbers primes) {
        BigInteger[] result = new BigInteger[9];
        for (int i = 0; i < 9; i++) {
            BigInteger val = coinValues[i];
            BigInteger divisor = primes.getLowestPrimeFactor(val);
            if (divisor.equals(val)) {
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
        int nCoinsSeen = 0;
        int nCoinsPossible = (int) Math.pow(2, Math.max(0, coinLength - 2));
        int i = 0;
        while (true) {
            if (nCoinsSeen >= nCoinsPossible) {
                throw new RuntimeException("saw all possible coins but didn't get enough to return!");
            }
            boolean[] coin = getRandomCoin(coinLength);
            String s = convertCoinToString(coin);
            boolean coinSeen = seenCoins.contains(s);  // TODO: make trie structure to store coins rather than linearly searching list
            if (coinSeen) continue;
            seenCoins.add(s);
            nCoinsSeen++;
            BigInteger[] values = getBaseValues(coin);
            BigInteger[] divisors = getNonTrivialDivisors(values, primes);
            if (divisors == null) continue;
            for (BigInteger divisor : divisors) {
                s += String.format(" %d", divisor);
            }
            
            System.out.println(i + ". found new coin: " + s);
            result[i] = s;
            i++;
            if (i >= nCoins) break;
        }
        return result;
    }

    public static void main(String[] args) {
        PrimeNumbers primes = new PrimeNumbers();
        int coinLength = Integer.parseInt(args[0]);
        int nCoins = Integer.parseInt(args[1]);
        String[] output = getOutput(coinLength, nCoins, primes);
    }
}