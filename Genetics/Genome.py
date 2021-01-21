class Genome:
    def __init__(self, string):
        string_chars = set(string)
        valid_chars = set("ACGTURYKMSWBDHVN")  # http://www.hgmd.cf.ac.uk/docs/nuc_lett.html
        assert string_chars - valid_chars == set(), "string contains invalid nucleotides: {}".format(string_chars)
        self.string = string

    @staticmethod
    def from_nih_file(fp):
        with open(fp) as f:
            lines = f.readlines()
        filename = fp.split("/")[-1]
        assert lines[0].split()[0].replace(">","") == filename.replace(".txt",""), "mismatch in header line and filename for {}".format(fp)
        s = ""
        for l in lines[1:]:  # don't put header in the genome
            s += l.strip()
        return Genome(s)

