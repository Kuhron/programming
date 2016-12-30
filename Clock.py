k = 3
wid = 72
chars = [" ", "."]

def display_binary_time(with_number = False):
    s = "{0:b}".format(math.floor(1000*time.time()))
    s = "0"*(k*math.ceil(wid/k)-len(s))+s
    s = "|"+"|".join([s[k*u:k*u+k] for u in range(math.ceil(len(s)/k))])+"|"
    s = "|"+"|".join([s[k*(k+1)*u:k*(k+1)*u+k*(k+1)] for u in range(math.ceil(len(s)/(k*(k+1))))])
    s_display = s.replace("0",chars[0]).replace("1",chars[1])
    if with_number:
        t_lst = []
        s_split = s.split("||")[1:-1]
        for u in range(len(s_split)):
            ss = s_split[u].split("|")
            for e in range(k):
                t_lst.append(ss[e])
        t = "".join([str(int(x, 2)) for x in t_lst])
        t_display = "|"+"|".join([t[k*u:k*u+k] for u in range(math.ceil(len(t)/k))])+"|"
        print(s_display + "  :  " + t_display)
    else:
        print(s_display)

# for k = 3, the fourth cell from the right lasts 4 seconds
# adjacent cells differ in period by a factor of 8 (=2**3)
# adjacent macro-cells differ by a factor of 512 (=8**3)
# so the cell cycle durations are as follows (assuming k = 3)
# the cell tick length is the length of the cell cycle before it
# the length of the macro-cell cycle is the length of its longest cell cycle
#--- MACRO-CELL -1 : 1/2 second
# 1/128 second
# 1/16 second
# 1/2 second
#--- MACRO-CELL -2 : 4 minutes
# 4 seconds
# 32 seconds
# 4 minutes 16 seconds
#--- MACRO-CELL -3 : 36 hours
# 34 minutes 8 seconds
# 4 hours 33 minutes
# 1 day 12 hours
#--- MACRO-CELL -4 : 2 years
# 12 days 3 hours
# 3 months 6 days
# 2 years 2 months
#--- MACRO-CELL -5 : 1000 years
# 17 years
# 136 years
# 1089 years (Middle Ages)
#--- MACRO-CELL -6 : 500,000 years
# 8711 years (Neolithic)
# 70 ka (Toba eruption)
# 557 ka (Pleistocene)
#--- MACRO-CELL -7 : 300,000,000 years
# 4.5 Ma (Pliocene (Neogene Period), first hominins)
# 36 Ma (Paleogene Period)
# 285 Ma (Permian Period)
#--- MACRO-CELL -8 : 150,000,000,000 years (full clock length)
# 2.3 Ga (early Proterozoic Eon, first eukaryotes)
# 18 Ga (before the Big Bang)
# 146 Ga (k = 3, wid = 72)
#---

if __name__ == "__main__":
    import math, time

    kuhoehjm = input("Show number? (y/n): ")
    user_choice_with_number = kuhoehjm == "y"
    
    while True:
        display_binary_time(with_number = user_choice_with_number)
        #time.sleep(1)
