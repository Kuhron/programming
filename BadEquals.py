class BadNumber:
    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        if type(other) is int:
            res = self.n == other
        elif type(other) is BadNumber:
            res = self.n == other.n
        else:
            return NotImplemented
        self.n += 1
        return res


if __name__ == "__main__":
    a = BadNumber(1)
    print(a == 1 and a == 2 and a == 3)


