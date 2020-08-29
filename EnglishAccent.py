import csv


dictionary_fp = "cmudict.txt"
conversion_table_fp = "Arpabet.csv"
output_fp = "EnglishAccentOutput.txt"



def read_conversion_table():
    # just returns lists for each row, so you can do with them as you please
    result = []
    with open(conversion_table_fp) as f:
        reader = csv.reader(f)
        for row in reader:
            result.append(row)
    return result


def convert_dictionary(arpabet_to_ipa):
    with open(dictionary_fp) as f:
        lines = [line.strip().split() for line in f.readlines()]
    d = {}
    arpabet_symbols_used = set()
    for line in lines:
        # combine word and its homograph index into one string
        key = line[0] + "-" + line[1]
        arpabet_symbols = line[2:]
        ipa_str = ""
        for symbol in arpabet_symbols:
            if symbol[-1] in "012":  # stress
                diacritic = arpabet_to_ipa[symbol[-1]]
                symbol = symbol[:-1]
            else:
                diacritic = ""
            arpabet_symbols_used.add(symbol)
            val = arpabet_to_ipa[symbol]
            ipa_symbol = val[0] + diacritic + val[1:]
            ipa_str += ipa_symbol
        d[key] = ipa_str
    with open(output_fp, "w") as f:
        for key in sorted(d.keys()):
            f.write("{} {}\n".format(key, d[key]))

    unused = [k for k in arpabet_to_ipa.keys() if k not in arpabet_symbols_used and k not in "012"]
    if len(unused) != 0:
        print("unused symbols: " + " ".join(sorted(unused)))

    return d


def convert_user_input(pronunciation_dict):
    while True:
        inp = input("Type something here to get the pronunciation:\n")
        inp = inp.strip().upper()
        result = ""
        for word in inp.split():
            options = [v for k, v in pronunciation_dict.items() if "".join(k.split("-")[:-1]) == word]
            if len(options) == 0:
                result += "[NotFound:{}] ".format(word)
            elif len(options) == 1:
                result += options[0] + " "
            else:
                result += "[" + "/".join(options) + "] "
        print(result + "\n")

    


if __name__ == "__main__":
    table = read_conversion_table()
    arpabet_index = table[0].index("Arpabet")
    other_options = [x for x in table[0] if x != "Arpabet"]
    print("Choose an accent:\n" + "\n".join("{}. {}".format(i, x) for i, x in enumerate(other_options)))
    inp = input()
    try:
        accent = other_options[int(inp)]
    except:
        raise Exception("failure")
    accent_index = table[0].index(accent)
    arpabet_to_ipa = {row[arpabet_index]: row[accent_index] for row in table[1:]}
    d = convert_dictionary(arpabet_to_ipa)
    convert_user_input(d)
