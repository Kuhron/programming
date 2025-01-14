import sys


def get_lines(fp, strip=False):
    with open(fp) as f:
        lines = f.readlines()

    if strip:
        lines = [l.strip() for l in lines]
    return lines


def get_line_to_indices(lines):
    d = {}
    for i,l in enumerate(lines):
        if l not in d:
            d[l] = []
        d[l].append(i)
    return d


def get_index_to_line(lines):
    return dict(enumerate(lines))


if __name__ == "__main__":
    fp = sys.argv[1]
    print(f"{fp = !r}")

    lines_to_omit = [
        "", "%", "{", "}", "TODO", "\\hline", "\\vfill",
    ] + [f"\\{be}{{{x}}}" for be in ["begin", "end"] for x in ["figure", "exe", "frame", "itemize", "center", "tabular", "table"]]

    strip = True

    lines = get_lines(fp, strip=strip)
    line_to_indices = get_line_to_indices(lines)
    index_to_line = get_index_to_line(lines)
    indices_strs = []
    line_strs = []
    for i, line in sorted(index_to_line.items()):
        indices_of_line = line_to_indices[line]
        if len(indices_of_line) > 1 and indices_of_line[0] == i and line not in lines_to_omit:
            max_to_show = 3
            if len(indices_of_line) <= max_to_show:
                indices_str = str(indices_of_line)
            else:
                indices_str = "[" + ", ".join(str(x) for x in indices_of_line[:max_to_show]) + f", ...] ({len(indices_of_line)})"
            indices_strs.append(indices_str)
            line_strs.append(line)

    m = max(len(x) for x in indices_strs)
    for indices_str, line_str in zip(indices_strs, line_strs):
        print(f"indices {indices_str.ljust(m)} | {line_str!r}")
