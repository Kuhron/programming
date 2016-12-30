class Hourglass {
	public static void main(String[] args) {
	    int START = 1;
	    int END = Integer.parseInt(args[0]);

	    for (int i=START; i<=END; i++) {
	        for (int j=START; j<=END; j++) {
                if (i==j || i==END-j+1 || (j%2==1 && (i==1 || i==END))) {
                    System.out.print((END-j+1) % 10);
                } else {
                    System.out.print(" ");
                }
            }

            System.out.println();
	    }
	}
}