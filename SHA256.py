import math

def string_to_8bit(s):
    result = ""
    for i in s:
        si = str(bin(ord(i)))[2:]
        si = "0"*(8-len(si)) + si
        result += si
    return result

def pad(s):
    if type(s) == str:
        m = string_to_8bit(s)
    else:
        m = str(bin(s))[2:]
    l = len(m)
    m += "1"
    k = (448-l-1) % 512
    #print(k)
    m += "0"*k
    len_64 = str(bin(l))
    len_64 = "0"*(66-len(len_64)) + len_64[2:]
    #print(len(len_64))
    m += len_64
    #print(m, len(m))
    return m

def parse(s):
    s = pad(s)
    result = []
    if len(s) % 512 != 0:
        print("function parse: function pad's output length mod 512 != 0")
        return result
    num_blocks = int(len(s)/512)
    for i in range(num_blocks):
        block = []
        for j in range(16):
            block.append(s[512*i+32*j:512*i+32*(j+1)])
        result.append(block)
    return result

def divisor_present(number, lst):
    for i in lst:
        if number % i == 0:
            return True
    return False

def get_primes(n):
    result = [2]
    i = 3
    while len(result) < n:
        if not divisor_present(i, result):
            result.append(i)
        i += 1
    return result

primes_64 = get_primes(64)

def get_word(flt):
    q = flt - int(flt)
    return int(q*(2**32))

hash_0 = [get_word(math.sqrt(i)) for i in primes_64[:8]]
k_series = [hex(get_word(i ** (1.0/3))) for i in primes_64]

# def encrypt_old(s):
#     initial_hash = hash_0
#     for message_block in parse(s):
#         hash_i = (hash_i + compress(hash_i)) % (2**32)
#         initial_hash = hash_i
#     return hash_i # hash_N

def encrypt(s):
    initial_hash = hash_0
    for i in range(len(parse(s))):
        a,b,c,d,e,f,g,h = [u for u in initial_hash]
        print(a,b,c,d,e,f,g,h)

# def compress(h):
#     pass

def shift(s, n):
    return str(int(s) >> n)

def rotate(s, n):
    b = str(bin(int(s)))[2:]
    return str(int(b[n:] + b[:n], 2))

def plus(lst):
    return str(bin((sum([int(e) for e in lst]))%(2**32)))

def ch(x,y,z):
    return plus([(x^y),(~x^z)])

def maj(x,y,z):
    return plus([(x^y),(x^z),(y^z)])

def SIGMA_0(x):
    return plus([rotate(x,2),rotate(x,13),rotate(x,22)])

def SIGMA_1(x):
    return plus([rotate(x,6),rotate(x,11),rotate(x,25)])

def sigma_0(x):
    return plus([rotate(x,7),rotate(x,18),shift(x,3)])

def sigma_1(x):
    return plus([rotate(x,17),rotate(x,19),rotate(x,10)])

def get_w(block):
    w = []
    for j in range(16):
        w.append(block[j])
    for j in range(16,64):
        w_ = sigma_1(w[j-2]) + w[j-7] + sigma_0(w[j-15]) + w[j-16]
        w.append(w_)
    return w
print(get_w(parse("abc")[0]))


print(encrypt("abc"))
print(encrypt("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))