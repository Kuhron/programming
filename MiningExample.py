import random
from hashlib import sha256


def increment_bytes(b):
    # little-endian for simplicity so we traverse the list in order instead of reverse
    res = bytearray()  # bytearray is mutable, bytes is not; like list and tuple
    carry = 1  # require adding one to the least significant bit in order to increment
    for byte in b:
        new_byte = byte + carry
        if new_byte >= 256:
            new_byte -= 256
            carry = 1
        else:
            carry = 0
        res.append(new_byte)

    # if there is still a carry, add new digit
    if carry == 1:
        res.append(0)
    return res


def test_increment_bytes():
    test_cases = [
        [[254, 1], [255, 1]],
        [[255, 1], [0, 2]],
        [[255, 255, 255, 13, 80], [0, 0, 0, 14, 80]],
        [[1, 2, 3], [2, 2, 3]],
        [[], [0]],
        [[255], [0, 0]],
        [[255, 255, 255, 255], [0, 0, 0, 0, 0]],
    ]
    for a, b in test_cases: 
        a = bytearray(a)
        b = bytearray(b)
        aa = increment_bytes(a)
        assert aa == b, f"expected 1 + {a} == {b} but got {aa}"
    print("passed test_increment_bytes")


def hash_to_bytes(b):
    return sha256(b).digest()


def bytes_starts_with_n_zero_bits(b, n):
    bytes_to_take, bits_left = divmod(n, 8)  # 256 == 2**8
    if bytes_to_take == len(b):
        return all(byte == 0 for byte in b)
    elif bytes_to_take > len(b):
        return False

    expect_zero_bytes = b[:bytes_to_take]
    next_byte = b[bytes_to_take]
    for byte in expect_zero_bytes:
        if byte != 0:
            return False

    assert 0 <= bits_left < 8
    return number_starts_with_n_zeros(next_byte, bits_left)


def number_starts_with_n_zeros(number, n):
    # in binary, and assumes we are looking at 8 bits
    assert 0 <= n < 8
    if type(number) is not int:
        raise TypeError(number)
    if number < 0:
        raise ValueError(number)
    if number >= 2 ** 8:
        print(f"Warning: number_starts_with_n_zeros called with too large of a number: {number}")
        return False
    n_bits_allowed_to_be_nonzero = 8 - n  # e.g. starts with 5 zeros, allowed to be 00000111 at most, which is 7 = 2 ** 3 - 1
    max_value = -1 + 2 ** n_bits_allowed_to_be_nonzero
    return number <= max_value


def get_n_leading_zero_bits_of_bytes(b):
    res = 0
    for byte in b:
        if byte == 0:
            res += 8
        else:
            n = get_n_leading_zero_bits_of_number(byte)
            res += n
            break  # don't add any more zeros
    return res


def get_n_leading_zero_bits_of_number(n):
    powers = range(8)
    z = 0
    for p in powers[::-1]:
        if n < 2**p:
            z += 1
    return z


def get_ascending_bytearrays():
    b = bytearray()
    while True:
        yield b
        b = increment_bytes(b)


def get_random_salt(n_bytes):
    return bytearray(random.randrange(256) for i in range(n_bytes))



if __name__ == "__main__":
    test_increment_bytes()
    salt = get_random_salt(n_bytes=8)
    for b in get_ascending_bytearrays():
        b = salt + b
        h = hash_to_bytes(b)
        z = get_n_leading_zero_bits_of_bytes(h)
        if z >= 20:
            print(f"{z} leading zeros: {list(b)} --> {list(h)}")

