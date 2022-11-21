import numpy as np
import matplotlib.pyplot as plt


def get_square_matrices(n, k):
    # all n*n matrices where elements are in Z_k
    for v in get_vectors(n, k):
        a = np.array(v).reshape((n, n))
        yield a


def get_vectors(n, k):
    # get vectors of length n**2 where elements are in Z_k
    v = [0] * n**2
    while True:
        yield v
        for i in range(len(v)+1):
            if i == len(v):
                # reached the end of the vector without being able to carry
                # can't get any more vectors, we've reached the max
                return
            x = v[-(i+1)]
            if x == k-1:
                # carry to next digit
                v[-(i+1)] = 0
            else:
                v[-(i+1)] += 1
                break


def mult(m1, m2, k):
    return (m1 @ m2) % k


def get_matrix_index(m, k, mats=None):
    if mats is None:
        # assuming the matrices are in the correct order
        # reshape/flatten is row-major
        m = m.flatten()  # not in-place
        N = m.size
        s = 0
        for p in range(0, N):
            c = m[N-p-1]
            s += c * k**p
        return s
    else:
        for i, m2 in enumerate(mats):
            if (m == m2).all():
                return i
        return None


def make_multiplication_table(mats, k):
    l = len(mats)
    a = np.zeros((l, l))
    for i in range(l):
        if i % 10 == 0:
            print(f"{i}/{l}")
        for j in range(l):  # not commutative
            mi = mats[i]
            mj = mats[j]
            m = mult(mi, mj, k)
            index = get_matrix_index(m, k, mats=mats)
            a[i,j] = index
    return a


def det(m, k):
    return int(round(np.linalg.det(m))) % k



if __name__ == "__main__":
    k = 3
    n = 2
    mats = list(get_square_matrices(n, k))
    det1 = [m for m in mats if det(m, k) == 1]
    for m in det1:
        print(m)

    # table = make_multiplication_table(mats, k)
    table = make_multiplication_table(det1, k)
    plt.imshow(table)
    plt.show()
