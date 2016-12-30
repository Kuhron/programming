import math, random, string, time

hash_length = 64

def l_to_n(l):
    return string.ascii_uppercase.index(l)

def n_to_l(n):
    return string.ascii_uppercase[n % 26]

def total_value(s):
    return sum([l_to_n(u) for u in s])

def viginere(s, key):
    result = ""
    for i in range(len(s)):
        result += n_to_l(l_to_n(s[i])+l_to_n(key[i % len(key)]))
    return result

def compound_viginere(s, number_of_times, separation):
    separation = 1 + (separation % len(s))
    for i in range(number_of_times):
        s = viginere(s[:-separation],s[separation:]) + viginere(s[-separation:],s[:separation])
    return s

def pad(s):
    s = s*math.ceil(float(hash_length/len(s)))
    return s[:hash_length]

# all nonsense strings used will be 64 characters
garble = pad(viginere("LOREMIPSUMDOLORSITAMETCLAUDIASEMPERFENESTRAMCLAUDITRACHOJMEDDAOX", "NONSEQUITURXMANKHWAAHAMGEREFT"))
salt = pad(viginere("ONCEUPONATIMEINALANDFARFARAWAYTHERELIVEDEIGHTEENWAITINEEDMORECHF", "OVERTHERIVERANDTHROUGHTHEWOODS"))
print(len(garble),len(salt))

# ------- HASHES ------- #

def short_viginere_hash(s):
    s = s.upper() + salt
    result = viginere(compound_viginere(s, 1+(total_value(s) % 100), l_to_n(s[0])), compound_viginere(garble, 1+(total_value(s) % 100), l_to_n(s[-1])))
    return result[:hash_length]

def long_viginere_hash(s):
    s = s.upper() + salt
    result = viginere(compound_viginere(s, total_value(s)), compound_viginere(garble, total_value(s)))
    return result[:hash_length]

def trivial_hash(s):
    return pad(s)

hash = short_viginere_hash

# --END-- HASHES --END-- #

def leading_char(s, c):
    i = 0
    while s[i] == c and i < len(s):
        i += 1
    return i

def test_sensitivity():
    print(hash("ALABASTERPEARLS"))
    print(hash("ARABASTERPEARLS"))
    print(hash("ALABASTERPARLS"))
    print(hash("ALABXASTERPEARLS"))
    print(hash("ALABASTERPEARLSWITHEXACTLYFORTYXXEIGHTCHARACTERS"))
    print(hash("ALABASTERPEARLSWITHEXACTLYFORTYXXEIGHTCHARACTERSANDTHENSOMEMORESTUFFTHATSHOULDAFFECTTHEOUTPUT"))
    print()

test_sensitivity()

def random_find(length, leading_a_required):
    start = time.time()
    time_dist = []
    while True:
        s = "".join([string.ascii_uppercase[math.floor(random.random()*26)] for i in range(length)])
        if leading_char(hash(s), "A") >= leading_a_required:
            print(s)
            print(hash(s))
            #return s
            time_taken = float(time.time()-start)
            time_dist.append(time_taken)
            print("Time taken to find this input: {0:.2f}\nAverage time taken: {1:.2f} ({2} samples)\n".format(time_taken, sum(time_dist)/len(time_dist), len(time_dist)))
            start = time.time()

def ordered_find(leading_a_required):
    a = 0
    # found = {}
    while True:
        xx = hash(n_to_s(a))
        if leading_char(xx, "A") >= leading_a_required:
            # if xx in found:
            #     found[xx]+=1
            # else:
            #     found[xx]=1
            print(xx, a, n_to_s(a))
        # else:
        #     print(a)
        a += 1

def n_to_s(n):
    if n == 0:
        return "A"
    result = []
    remainder = n
    power = math.floor(math.log(n, 26))
    while remainder > 0:
        taken = math.floor(remainder/(26**power))
        result.append(taken)
        remainder -= taken*(26**power)
        power -= 1
    if remainder == 0 and power >= 0:
        result.extend([0]*(power+1))
    return "".join([n_to_l(u) for u in result])

ordered_find(1)

# find(12,n) with hash = short_viginere_hash takes: {1: 0.78 (69 samples), 2: 19.80 (40 samples), 3: 284.98 (5 samples)}