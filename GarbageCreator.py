import random


def get_random_bytes(n):
    barray = bytearray(random.randrange(256) for i in range(n))
    return bytes(barray)


if __name__ == "__main__":
    # print(get_random_bytes(77))
    with open("FinalEssay.pdf", "wb") as f:
        f.write(get_random_bytes(100000))
