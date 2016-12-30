def c(m, n):
    for i in range(m, n):
        j = i
        k = 0
        while j != 1 and k<1000000:
            if j % 2 == 0:
                j = j/2
            else:
                j = 3*j+1
            k += 1
        if j == 1:
            #print("1 reached for value %d" % i)
            pass
        else:
            print("one million iterations reached for value %d" % i)
            sys.exit()
    #print("finished")

if __name__ == "__main__":
    import sys
    for h in range(4,10**50):
        print(10**6*h)
        c(10**6*h, 10**6*(h+1))
