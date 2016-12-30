def make_random_array(x,y):
    a = []
    for i in range(x):
        b = []
        for j in range(y):
            b.append(math.floor(random.random()*1000))
        a.append(b)
    return a

def print_array(a):
    for i in range(len(a)):
        for j in range(len(a[i])-1):
            print("%03d" % a[i][j], end = " ")
        print("%03d" % a[i][len(a[i])-1])

def place_barrier(a):
    num_cands = len(a)-1+len(a[0])-1
    c = random.randrange(num_cands)
    lesser = []
    greater = []
    if c < len(a)-1:
        print("vertical")
        for i in range(len(a)):
            bl = []
            bg = []
            for j in range(c+1):
                bl.append(a[i][j])
            lesser.append(bl)
            for j in range(c+1, len(a)):
                bg.append(a[i][j])
            greater.append(bg)    
    else:
        print("horizontal")
        #"""
        for i in range(c-(len(a)-1)+1):
            bl = []
            for j in range(len(a[0])):
                bl.append(a[i][j])
            lesser.append(bl)
        for i in range(c-(len(a)-1)+1, len(a)):
            bg = []
            for j in range(len(a[0])):
                bg.append(a[i][j])
            greater.append(bg)
        #"""
    return lesser, greater
            
def force(x,y):
    # force on x from y, positive means away from x
    # k should be subtracted from x and added to y
    if x > y:
        k = x-y#math.floor((x+y)/2)
    elif x < y:
        # let the greater one handle it so the pair is not double-counted
        k = 0#-math.floor((x+y)/2)
    else:
        k = 0
    return k

def is_in_equilibrium(a):
    return max_array(a)-min_array(a) <= 1

def max_array(a):
    result = a[0][0]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result = max(result, a[i][j])
    return result

def min_array(a):
    result = a[0][0]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result = min(result, a[i][j])
    return result

def how_many(a, val):
    count = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] == val:
                count += 1
    return count

def neighbors(a, i, j):
    result = []
    for ni in [-1,0,1]:
        for nj in [-1,0,1]:
            if (
                i+ni >= 0 and
                i+ni < len(a) and
                j+nj >= 0 and
                j+nj < len(a[0]) and
                [ni, nj] != [0,0]
                ):
                result.append([i+ni, j+nj])
    return result

def sorted_neighbors(a, i, j):
    # don't need it no more
    d = {}
    for e, u in neighbors(a, i, j):
        k = abs(a[i][j] - a[e][u])
        if k in d:
            d[k].append([e, u])
        else:
            d[k] = [[e, u]]
    return [d[k] for k in sorted(d)]

def step(a):
    b = [[a[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    transfers = [[0 for j in range(len(a[0]))] for i in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            n_lst = neighbors(a, i, j)
            transfers_ij = []
            val = a[i][j]
            total_force = float(sum([math.floor(force(val, a[m[0]][m[1]])) for m in n_lst]))
            #print(total_force)
            for n in n_lst:
                full_force = force(val, a[n[0]][n[1]])
                proportional_force = math.floor(val*full_force/max(total_force, 1.0))
                transfers_ij.append(proportional_force)
            for k in range(len(transfers_ij)):
                transfers[i][j] -= transfers_ij[k]
                n = n_lst[k]
                transfers[n[0]][n[1]] += transfers_ij[k]
    #print_array(transfers)
    for i in range(len(a)):
        for j in range(len(a[0])):
            b[i][j] += transfers[i][j]
    return b

def sum_array(a):
    result = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            result += a[i][j]
    return result

if __name__ == "__main__":
    import math
    import random
    import time

    a = make_random_array(5,5)
    """
    a = [
        [4,0,4],
        [0,2,0],
        [4,0,4]
    ]
    """
    print_array(a)
    print()
    """
    aa = [[0 for j in range(5)] for i in range(5)]
    for ii in range(5):
        for jj in range(5):
            a = [[0 for j in range(5)] for i in range(5)]
            qi = ii#random.randrange(len(a))
            qj = jj#random.randrange(len(a[0]))
            a[qi][qj] = 100
            #print_array(a)
    """
    #print("sum", sum_array(a))
    #lesser, greater = place_barrier(a)
    #print("Lesser:")
    #print_array(lesser)
    #print("Greater:")
    #print_array(greater)
    #print(is_in_equilibrium(a))
    steps = 0
    while not is_in_equilibrium(a):
        #time.sleep(0.5)
        b = step(a)
        if b == a:
            # no flow occurred, even though we are not in equilibrium
            print("Gradient became too weak to result in flow.")
            break
        a = b
        steps += 1
        print_array(a)
        print()
        #print("sum", sum_array(a))
            #aa[ii][jj] = steps
    print("steps:", steps)
    #print_array(aa)
    
