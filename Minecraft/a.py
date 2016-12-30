p = r"C:\Users\Wesley\AppData\Roaming\.minecraft\saves\PythonExperimentation\\"
fi = p + r"region\r.0.0.mca"
# fi = p + r"level.dat"

with open(fi, "rb") as f:
    b = f.read()

def f():
    i = 0
    e = len(b)
    while i < e:
        le = input("num bytes to print: ")
        try:
            le = int(le)
        except ValueError:
            le = 1
        v = b[i: i+le]
        yyy(v)
        i += le

def yyy(v):
    js = []
    for j in v:
        js.append("{0:08b}".format(j))
    print(",".join(js))

if __name__ == "__main__":
    f()