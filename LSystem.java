import java.util.Arrays;


class Rule {
    String from;
    String to;

    public Rule(String from, String to) {
        this.from = from;
        this.to = to;
    }

    public String applyOnce(String source) {
        String result = "";
        for (int i = 0; i < source.length(); i++) {
            String c = String.valueOf(source.charAt(i));
            if (c.equals(this.from)) {
                result += this.to;
            } else {
                result += c;
            }
        }
        return result;
    }

    public String apply(String source, int nTimes) {
        String result = source;
        for (int i = 0; i < nTimes; i++) {
            result = this.applyOnce(result);
        }
        return result;
    }
}


class RuleSet {
    static final String SEPARATOR = ":";

    Rule[] rules;

    public RuleSet(Rule[] rules) {
        this.rules = rules;
    }

    public static RuleSet fromRuleStrings(String[] strings) {
        System.err.println(Arrays.toString(strings));
        Rule[] result = new Rule[strings.length];
        for (int i = 0; i < strings.length; i++) {
            String s = strings[i];
            System.err.println("new string: " + s);  // why doesn't this print anything?
            String[] split = s.split(SEPARATOR);
            if (split.length != 2) {
                String errorMsg = String.format("%s does not contain the %s separator", s, SEPARATOR);
                throw new RuntimeException(errorMsg);
            }
            System.err.println("split: " + split);
            Rule rule = new Rule(split[0], split[1]);
            result[i] = rule;
        }
        return new RuleSet(result);
    }

    public String apply(String source, int nTimes) {
        String result = source;
        for (int i = 0; i < nTimes; i++) {
            for (Rule rule : this.rules) {
                result = rule.applyOnce(result);
            }
        }
        return result;
    }
}


class LSystem {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println(String.format("usage: java LSystem {source string} [{rules using %s as separator}]", RuleSet.SEPARATOR));
            System.exit(0);
        }
        String source = args[0];
        String[] ruleStrings = Arrays.copyOfRange(args, 1, args.length);
        RuleSet ruleSet = RuleSet.fromRuleStrings(ruleStrings);
        System.out.println(String.format("%s (%d chars)", source, source.length()));
        // String from = "A";
        // String to = "ABC";
        // Rule rule = new Rule(from, to);
        int nTimes = 5;
        // System.out.println(rule.apply(source, nTimes));
        System.out.println(ruleSet.apply(source, nTimes));
    }
}