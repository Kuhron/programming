import time

hms = input("input hours:minutes:seconds\n")
h,m,s = hms.split(":")
h = int(h)
m = int(m)
s = int(s)

dt_total = 3600*h + 60*m + s
t0 = time.time()
tf = t0 + dt_total

def seconds_to_str(s):
    h,rest = divmod(s, 3600)
    m,s = divmod(rest, 60)
    h = int(h)
    m = int(m)
    s = int(s)
    return f"{h}:{m:02d}:{s:02d}"


while True:
    t = time.time()
    dt = t - t0
    if t >= tf:
        print("timer complete")
        break
    else:
        t_left = tf - t
        p = dt/dt_total
        dt_str = seconds_to_str(dt)
        rem_str = seconds_to_str(t_left)
        print(f"{rem_str} remaining, {dt_str} elapsed ({p:.2%})")
        time.sleep(0.5)
