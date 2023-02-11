import random
import itertools

# fams_countries = pd.read_excel("LanguageSorting.ods", engine="odf", sheet_name="FamiliesCountries", header=0, index_col=0)
fams_countries_fp = "fams_countries.tsv"
with open(fams_countries_fp) as f:
    lines = f.readlines()
true_fams_by_country = {}
distractor_fams_by_country = {}
true_countries_by_fam = {}
distractor_countries_by_fam = {}
lines_arr = [l.replace("\n", "").split("\t") for l in lines]
header = lines_arr[0]
row_labels = [l[0] for l in lines_arr]
n_rows = len(row_labels) - 1
n_cols = len(header) - 1
countries = row_labels[1:]
fams = header[1:]
for i in range(1, n_rows):
    country = row_labels[i]
    assert country not in true_fams_by_country
    true_fams_by_country[country] = []
    assert country not in distractor_fams_by_country
    distractor_fams_by_country[country] = []
    for j in range(1, n_cols):
        lg = header[j]
        val = lines_arr[i][j]
        if lg not in true_countries_by_fam:
            true_countries_by_fam[lg] = []
        if lg not in distractor_countries_by_fam:
            distractor_countries_by_fam[lg] = []
        if val == "x":
            true_fams_by_country[country].append(lg)
            true_countries_by_fam[lg].append(country)
        elif val == "d":
            distractor_fams_by_country[country].append(lg)
            distractor_countries_by_fam[lg].append(country)
        else:
            pass
# print("\ntfc:", true_fams_by_country)
# print("\ndfc:", distractor_fams_by_country)
# print("\ntcf:", true_countries_by_fam)
# print("\ndcf:", distractor_countries_by_fam)


# langs_fams = pd.read_excel("LanguageSorting.ods", engine="odf", sheet_name="LanguagesFamilies", header=None)
langs_fams_fp = "langs_fams.txt"
with open(langs_fams_fp) as f:
    langs_fams = [l.strip() for l in f.readlines()]
langs_fams_lists = [s.split(" < ") for s in langs_fams]
langs = [l[0] for l in langs_fams_lists]  # the actual languages, terminal nodes

descent_by_lang = {}
for l in langs_fams_lists:
    for i in range(len(l)):
        lg = l[i]
        descent = l[i:]
        if lg in descent_by_lang:
            assert descent_by_lang[lg] == descent, (lg, l)
        else:
            descent_by_lang[lg] = descent

fam_graph = {}
def add_to_fam_graph(a, b):
    if a in fam_graph:
        assert fam_graph[a] == b, (a, fam_graph[a], b)
    else:
        fam_graph[a] = b


for fams2 in langs_fams_lists:
    for x, y in zip(fams2[:-1], fams2[1:]):
        add_to_fam_graph(x, y)

members_by_family = {}
for fams2 in langs_fams_lists:
    # everything in the list is a member of itself and everything to its right
    for i in range(len(fams2)):
        smaller = fams2[i]
        for j in range(i, len(fams2)):
            larger = fams2[j]
            if larger not in members_by_family:
                members_by_family[larger] = set()
            members_by_family[larger].add(smaller)

def is_contained_in(x, y):
    if x == y:
        return True
    counter = 0
    while counter < 1000:
        if x not in fam_graph:
            return False
        elif fam_graph[x] == y:
            return True
        else:
            x = fam_graph[x]
            counter += 1
    raise RuntimeError

def make_family_containing_question_answer_true():
    counter = 0
    while counter < 1000:
        l = random.choice(langs_fams_lists)
        try:
            i,j = sorted(random.sample(list(range(len(l))), 2))
        except ValueError:
            counter += 1
            continue
        x = l[i]
        y = l[j]
        return f"{x} < {y}"
    raise RuntimeError

def make_family_containing_question_answer_false():
    all_things = set(fam_graph.keys()) | set(fam_graph.values())
    counter = 0
    while counter < 1000:
        x,y = random.sample(list(all_things), 2)
        if y in langs:
            # don't put terminal on the right side
            counter += 1
            continue
        if not is_contained_in(x, y):
            return f"{x} < {y}"
        counter += 1
    raise RuntimeError

def make_family_containing_question():
    res_is_true = random.random() < 0.5
    n_true = 1 if res_is_true else 3
    n_false = 4 - n_true
    answers = [make_family_containing_question_answer_true() for i in range(n_true)] + [make_family_containing_question_answer_false() for i in range(n_false)]
    a = answers[0 if res_is_true else -1]
    random.shuffle(answers)
    print(f"\nWhich of the following is {str(res_is_true).lower()}?")
    for x in answers:
        print(x)
    input("\npress enter to see answer")
    print(f"{str(res_is_true).lower()}: {a}")

def make_language_in_country_question_answer_true():
    counter = 0
    while counter < 1000:
        c = random.choice(countries)
        # row = fams_countries.loc[c, :]
        # trues = fams_countries.columns[row == "x"]
        trues = true_fams_by_country.get(c, [])
        if len(trues) < 1:
            counter += 1
            continue
        lg = random.choice(trues)
        return f"{lg} is in {c}"
    raise RuntimeError

def make_language_in_country_question_answer_false():
    counter = 0
    while counter < 1000:
        c = random.choice(countries)
        # row = fams_countries.loc[c, :]
        # distractors = fams_countries.columns[row == "d"]
        distractors = distractor_fams_by_country.get(c, [])
        if len(distractors) < 1:
            counter += 1
            continue
        lg = random.choice(distractors)
        return f"{lg} is in {c}"
    raise RuntimeError

def make_language_in_country_question():
    res_is_true = random.random() < 0.5
    n_true = 1 if res_is_true else 3
    n_false = 4 - n_true
    answers = [make_language_in_country_question_answer_true() for i in range(n_true)] + [make_language_in_country_question_answer_false() for i in range(n_false)]
    a = answers[0 if res_is_true else -1]
    random.shuffle(answers)
    print(f"\nWhich of the following is {str(res_is_true).lower()}?")
    for x in answers:
        print(x)
    input("\npress enter to see answer")
    print(f"{str(res_is_true).lower()}: {a}")

def make_language_in_single_country_question():
    counter = 0
    while counter < 1000:
        c = random.choice(countries)
        # row = fams_countries.loc[c, :]
        # trues = fams_countries.columns[row == "x"]
        # distractors = fams_countries.columns[row == "d"]
        trues = true_fams_by_country.get(c, [])
        distractors = distractor_fams_by_country.get(c, [])
        # need at least one true and at least one distractor, at least 4 total
        if len(trues) < 1 or len(distractors) < 1 or len(trues) + len(distractors) < 4:
            counter += 1
            continue
        answer_can_be_true = len(distractors) >= 3
        answer_can_be_false = len(trues) >= 3
        if not (answer_can_be_true or answer_can_be_false):
            counter += 1
            continue
        if not answer_can_be_true:
            res_is_true = False
        elif not answer_can_be_false:
            res_is_true = True
        else:
            res_is_true = random.random() < 0.5
        n_true = 1 if res_is_true else 3
        n_false = 4 - n_true
        answers = random.sample(list(trues), n_true) + random.sample(list(distractors), n_false)
        a = answers[0 if res_is_true else -1]
        is_s = "IS" if res_is_true else "is NOT"
        print(f"\nWhich of the following languages or families {is_s} in {c}?")
        random.shuffle(answers)
        for x in answers:
            print(x)
        input("\npress enter to see answer")
        print(f"{is_s} in {c}: {a}")
        return
    raise RuntimeError

def make_single_language_in_country_question():
    counter = 0
    while counter < 1000:
        lg = random.choice(fams)
        print(lg)
        print(fams)
        # col = fams_countries.loc[:, lg]
        # trues = fams_countries.index[col == "x"]
        # distractors = fams_countries.index[col == "d"]
        trues = true_countries_by_fam.get(lg, [])
        distractors = distractor_countries_by_fam.get(lg, [])
        # need at least one true and at least one distractor, at least 4 total
        if len(trues) < 1 or len(distractors) < 1 or len(trues) + len(distractors) < 4:
            counter += 1
            continue
        answer_can_be_true = len(distractors) >= 3
        answer_can_be_false = len(trues) >= 3
        if not (answer_can_be_true or answer_can_be_false):
            counter += 1
            continue
        if not answer_can_be_true:
            res_is_true = False
        elif not answer_can_be_false:
            res_is_true = True
        else:
            res_is_true = random.random() < 0.5
        n_true = 1 if res_is_true else 3
        n_false = 4 - n_true
        answers = random.sample(list(trues), n_true) + random.sample(list(distractors), n_false)
        a = answers[0 if res_is_true else -1]
        is_s = "DOES" if res_is_true else "does NOT"
        print(f"\nWhich of the following countries {is_s} contain {lg}?")
        random.shuffle(answers)
        for x in answers:
            print(x)
        input("\npress enter to see answer")
        print(f"{is_s} contain {lg}: {a}")
        return
    raise RuntimeError

def make_language_family_containing_languages_question():
    # the prompt is three terminal nodes, and they have to say which is the smallest family containing them
    potential_families = [x for x, members in members_by_family.items() if len([lg for lg in members if lg in langs]) >= 3]
    fam = random.choice(potential_families)
    lgs = random.sample([lg for lg in members_by_family[fam] if lg in langs], 3)
    potential_matching_fams = [f for f, members in members_by_family.items() if all(lg in members for lg in lgs)]
    sizes = [len(members_by_family[f]) for f in potential_matching_fams]
    min_size = min(sizes)
    assert sizes.count(min_size) == 1, potential_matching_families
    fam = potential_matching_fams[sizes.index(min_size)]
    print("\nWhat is the smallest family or branch of a family containing the following languages?")
    print(", ".join(lgs))
    input("\npress enter to see answer")
    print(f"answer: {fam}")
    print("the individual languages are:")
    for lg in lgs:
        print(' < '.join(descent_by_lang[lg]))

def make_odd_language_out_question():
    # choose three languages in the same smaller group, and one in some other group
    # maybe should make family distractors?
    counter = 0
    while counter < 1000:
        potential_families = [x for x, members in members_by_family.items() if len([lg for lg in members if lg in langs]) >= 3]
        fam = random.choice(potential_families)
        lgs = random.sample([lg for lg in members_by_family[fam] if lg in langs], 3)
        potential_matching_fams = [f for f, members in members_by_family.items() if all(lg in members for lg in lgs)]
        sizes = [len(members_by_family[f]) for f in potential_matching_fams]
        min_size = min(sizes)
        assert sizes.count(min_size) == 1, potential_matching_families
        fam = potential_matching_fams[sizes.index(min_size)]
        # if there are other potential matching families, choose another language from one of those as the distractor
        other_fams = [potential_matching_fams[i] for i in range(len(sizes)) if sizes[i] != min_size]
        langs_in_other_fams = set()
        for f in other_fams:
            langs_in_other_fams |= members_by_family[f]
        langs_in_other_fams = [x for x in langs_in_other_fams if x not in lgs]
        if len(langs_in_other_fams) < 1:
            if random.random() < 0.33:
                # pick a small other family
                wrong_fams = [f for f, members in members_by_family.items() if len(members) <= 3]
                fam2 = random.choice(wrong_fams)
            else:
                counter += 1
                continue
        else:
            msize2 = min(s for s in sizes if s != min_size)
            weights = [(0 if s == min_size else 1/s) for s in sizes]
            wt = sum(weights)
            weights = [w/wt for w in weights]
            fam2 = weighted_choice(potential_matching_fams, weights)
        # print(fam, members_by_family[fam])
        # print(fam2, members_by_family[fam2])
        try:
            a = random.choice([x for x in members_by_family[fam2] if x in langs and x not in members_by_family[fam]])
        except IndexError:
            counter += 1
            continue
        answers = lgs + [a]
        random.shuffle(answers)
        print("\nWhich of these languages is least related to the others?")
        for x in answers:
            print(x)
        input("\npress enter to see answer")
        print(f"answer: {a}\n({' < '.join(descent_by_lang[a])})\n(the others are {' < '.join(descent_by_lang[fam])})")
        return
    raise RuntimeError

def weighted_choice(xs, ws):
    assert len(xs) == len(ws)
    assert abs(sum(ws) - 1) < 1e-6, "weights must sum to 1"
    r = random.random()
    t = 0
    ts = []
    for w in ws:
        t += w
        ts.append(t)
    for i, t in enumerate(ts):
        if t >= r:
            return xs[i]
    raise RuntimeError


last = None
while True:
    fs = [make_family_containing_question, make_language_in_country_question, make_language_in_single_country_question, make_single_language_in_country_question, make_language_family_containing_languages_question, make_odd_language_out_question]
    ss = ["family containment", "languages in countries", "languages in single country", "single language in countries", "language family containing languages", "odd language out"]
    print("\n--------\nChoose the question type you want" + (f" (or just press enter to use the previous choice, which is {last})" if last is not None else "") + ":")
    for i in range(len(fs)):
        print(f"{i}. {ss[i]}")
    choice = input()
    if choice.strip() == "" and last is not None:
        choice = last
    else:
        try:
            choice = int(choice.strip())
            last = choice
        except ValueError:
            print("invalid choice")
            continue
    f = fs[choice]
    f()
    input("\npress enter to continue")



