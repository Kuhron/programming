import java.lang.Math;
import java.util.ArrayList; 
import java.util.Arrays;
import java.util.List;


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
            result += (Math.pow(base, power));
        }
        return result;
    }

    public static void main(String[] args) {
        PrimeNumbers primes = new PrimeNumbers();
        int n = 49;
        int p = primes.getLowestPrimeFactor(n);
        System.out.println(p);
    }
}