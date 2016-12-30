import math, msvcrt, random, time

# ------- useless functions ------- #

# def do_nothing():
#     pass

# def demonstrate_carriage_return():
#     for i in range(10**6):
#         print(i, end = "\r")

# --END-- useless functions --END-- #

# ------- misc ------- #

def get_parabolic_price_trajectory(p_initial, v_initial, a):
    if a > 0:
        a = -a
    if v_initial < 0:
        v_initial = -v_initial
    if p_initial < 0:
        p_initial = -p_initial
    p = p_initial
    v = v_initial
    result = []
    while p >= p_initial:
        result.append(math.floor(p))
        p += v
        v += a
    # prevent user from detecting critical point because two consecutive values are the same
    indices_to_remove = []
    for i in range(len(result)-1):
        if result[i] == result[i+1]:
            indices_to_remove.append(i)
    if indices_to_remove != []:
        result = result[:min(indices_to_remove)] + result[max(indices_to_remove)+1:]
    return result
#print(get_price_trajectory(4,15,-2))

def get_normal_variate_price_trajectory(p_initial, v_base, a):
    # treat a as mass or something, attenuate the variance of the distribution around the mean of mu_v
    if a > 0:
        a = -a
    
    # treat the parameters of v as the source of the position shocks; i.e., change p each time by a random draw from the distribution of v
    # v_ = random.normalvariate(mu_v_, sigma_v_) # can be less than zero, that's what the fun's all about
    
    if p_initial < 0:
        p_initial = -p_initial
    p = p_initial
    
    result = []
    len_result = 0
    
    while p >= 0 and len_result < 100:
        result.append(math.floor(p+0.5))
        len_result += 1
        p += random.normalvariate(0, v_base)/a
    return result

def display_prices(lst):
    strlst = []
    for p in lst:
        strp = str(p)
        strlst.append(" "*(6-len(strp)) + strp)
    print(" ".join(strlst), end = "\r")

def gpt_dist(mu_p, sigma_p, mu_v, sigma_v, mu_a, sigma_a):
    #f = random.choice([get_parabolic_price_trajectory, get_normal_variate_price_trajectory])
    f = get_normal_variate_price_trajectory
    return f(random.normalvariate(mu_p, sigma_p), max(random.normalvariate(mu_v, sigma_v),1), max(random.normalvariate(mu_a, sigma_a),1))

def step(cq):
    for cell in range(len(cq)):
        if cq[cell] == [] and random.random() < 0.1:
            cq[cell] = gpt_dist(100,100,20,20,-3,2)
        elif cq[cell] != []:
            cq[cell] = cq[cell][1:]

def display_cell_queue(cq, owned):
    qq = []
    for q in range(len(cq)):
        if cq[q] == []:
            qq.append("")
        elif q in owned:
            qq.append("["+str(cq[q][0])+"]")
        else:
            qq.append(" "+str(cq[q][0])+" ")
    display_prices(qq)

# --END-- misc --END-- #

# shows you a good estimate for average length of price trajectories
# ehoc = []
# for e in range(10**5):
#     ehoc.append(len(gpt_dist(100,100,20,20,-3,2)))
# print(float(sum(ehoc))/len(ehoc))
# import sys
# sys.exit()

def pad(s, c, n, front = True):
    s = str(s)
    c = str(c)
    q = n-len(s)
    if front:
        return c*q+s
    else:
        return s+c*q

if __name__ == "__main__":
    k = 5
    delay = 0.5

    cell_queue = [[] for i in range(k)]
    # for a in gpt_dist(100,100,20,20,-3,2):
    #     print(a, end = "\r")
    #     time.sleep(0.8)

    owned = []
    money = 0
    t0 = time.time()
    total_ticks = 1
    print(" ".join([" "*3+str(u)+" "*2 for u in range(k)]))
    while True:
        while time.time() - t0 < delay:
            if msvcrt.kbhit():
                prop = ord(msvcrt.getch())-48
                #print(prop, "-"*4)
                # if prop == "\r": # not sure what this part does, got it from StackOverflow #19508353
                #     break
                if prop in range(k):
                    if prop not in owned:
                        if cell_queue[prop] != []:
                            money -= cell_queue[prop][0]
                            owned.append(prop)
                    else:
                        money += cell_queue[prop][0]
                        owned.remove(prop)
                        cell_queue[prop] = cell_queue[prop][:1]
        t0 = time.time()
        #thing_to_display = cell_queue + [["____ ${0}".format(pad(money," ",7))]]
        thing_to_display = cell_queue + [["____ ${0}, average ${1:.0f} per tick".format(pad(money," ",7), float(money)/total_ticks)]]
        display_cell_queue(thing_to_display, owned)
        step(cell_queue)
        indices_to_remove = []
        for i in range(len(owned)):
            if cell_queue[owned[i]] == []:
                indices_to_remove.append(i)
        total_ticks += 1

        # NOTE: if you do not sell a property by the time it disappears, you do not get any money back and your entire purchase cost is sunk!
        owned_new = []
        for i in range(len(owned)):
            if i not in indices_to_remove:
                owned_new.append(owned[i])
        owned = owned_new