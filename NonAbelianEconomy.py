import random, string

k = 26

ordering = string.ascii_uppercase[:k]
initial_wallet = ordering #ordering[::-1]

def rate(wallet):
    # this function may be changed to yield a differently-functioning economy
    result = 0
    for i in range(len(wallet)-1):
        if ordering.index(wallet[i]) < ordering.index(wallet[i+1]):
            result += 1
        else:
            pass #result += -1
    return result

def random_transformation():
    s = [i for i in range(k)]
    random.shuffle(s)
    return s

def transform(s, transformation):
    result = ""
    for position in transformation:
        result += s[position]
    return result

def worth_change(wallet, transformation):
    a = rate(wallet)
    b = rate(transform(wallet, transformation))
    return b - a

def invert(transformation):
    result = []
    for i in range(len(transformation)):
        result.append(transformation.index(i))
    return result

class Person:
    def __init__(self, name, verbose = True):
        self.name = name
        self.wallet = transform(ordering, random_transformation()) #initial_wallet
        if verbose:
            print("{0} is a person with wallet {1}, worth {2}.".format(self.name, self.wallet, self.get_worth()))

    def get_worth(self):
        return rate(self.wallet)

    def allowed_to_buy(self):
        return (self.get_worth() > 0)

    def buy(self, transformation):
        self.wallet = transform(self.wallet,invert(transformation))

    def sell(self, transformation):
        self.wallet = transform(self.wallet,transformation)

class Good:
    def __init__(self, name, transformation, verbose = True):
        self.name = name
        self.price = transformation
        if verbose:
            print("{0} is a good with price {1}.".format(self.name, self.show_price()))

    def show_price(self):
        return ".".join([str(i) for i in self.price])

def trade(buyer, seller, good, verbose = True):
    if not buyer.allowed_to_buy():
        if verbose:
            print("{0}'s wallet ({1}) is worth too little ({2})!".format(buyer.name, buyer.wallet, buyer.get_worth()))
        return None
    buyer.buy(good.price)
    seller.sell(good.price)
    if verbose:
        print("{0} has bought a {4} for {5} from {2}.\n{0} now has wallet {6}, worth {1}.\n{2} now has wallet {7}, worth {3}.".format(buyer.name, buyer.get_worth(), seller.name, seller.get_worth(), good.name, good.show_price(), buyer.wallet, seller.wallet))

def worth_change_by_trade(person, good, role):
    # role = "buyer" or "seller"
    if role == "buyer":
        return worth_change(person.wallet, invert(good.price))
    elif role == "seller":
        return worth_change(person.wallet, good.price)
    else:
        print("function trade_makes_sense_to: invalid role passed")
        return None

def total_worth_change(buyer, seller, good):
    return worth_change_by_trade(buyer, good, "buyer") + worth_change_by_trade(seller, good, "seller")

def trade_makes_sense_to(person, good, role):
    return (worth_change_by_trade(person, good, role) >= 0) # mess with >0, >=0, and whatever else

def trade_makes_sense(buyer, seller, good):
    return (trade_makes_sense_to(buyer, good, "buyer") and trade_makes_sense_to(seller, good, "seller"))

def main():
    Alice = Person("Alice")
    Bob = Person("Bob")
    # Book = Good("Book", random_transformation())
    # trade(Alice, Bob, Book)
    # trade(Bob, Alice, Book)
    # aw = Alice.wallet
    # bw = Bob.wallet
    # print(worth_change(aw, Book.price))
    # print(worth_change(aw, invert(Book.price)))
    # print(worth_change(bw, Book.price))
    # print(worth_change(bw, invert(Book.price)))

    ratio = 0.8
    while Alice.get_worth() < ratio*float(k) or Bob.get_worth() < ratio*float(k):
        Book = Good("Book", random_transformation(), False)
        if trade_makes_sense(Alice, Bob, Book) and not trade_makes_sense(Bob, Alice, Book):
            trade(Alice, Bob, Book)
        elif trade_makes_sense(Bob, Alice, Book) and not trade_makes_sense(Alice, Bob, Book):
            trade(Bob, Alice, Book)
        elif trade_makes_sense(Alice, Bob, Book) and trade_makes_sense(Bob, Alice, Book):
            a = total_worth_change(Alice, Bob, Book)
            b = total_worth_change(Bob, Alice, Book)
            if a > b:
                trade(Alice, Bob, Book)
            elif b > a:
                trade(Bob, Alice, Book)
            elif random.random() < 0.5:
                trade(Alice, Bob, Book)
            else:
                trade(Bob, Alice, Book)
        else:
            # no trade makes sense
            pass

if __name__ == "__main__":
    main()