f = lambda x: x+1
g = f.__get__(5)
print(g())  # 6

f = lambda x, y: x+y
g = f.__get__(5)
print(g(4))  # 9

f = lambda x, y: x+y
g = f.__get__(4, 3)  # __get__ has params (self, instance, owner), so instance is 4 and owner is set to something not correct (3) but also which is not used
print(g(11))  # 15; the 3 is not used, so this is not the same as Haskell partial application

print("\n----\n")


# descriptors, from stackoverflow question 3798835

class Celsius:
    def __get__(self, instance, owner):
        print(f"called __get__ with instance={instance} and owner={owner}")
        if instance is None:
            print("called static method, owner is", owner)
        else:
            print("called instance method")
            f = instance.fahrenheit
            return 5/9 * (f - 32)

    def __set__(self, instance, value):
        print(f"called __set__ with instance={instance} and value={value}")
        f = 32 + 9/5 * value
        instance.fahrenheit = f

class Temperature:
    celsius = Celsius()  # class variable but can also be accessed from instance

    def __init__(self, initial_f):
        self.fahrenheit = initial_f

t = Temperature(212)
print("t.celsius is", t.celsius)
print("Temperature.celsius is", Temperature.celsius)
t.celsius = 56
print("t.fahrenheit is now", t.fahrenheit)
print("t.celsius is now", t.celsius)

print("\n----\n")


# from stackoverflow question 20533349

class StaticOrInstanceDescriptor:
    def __init__(self, static):
        print(f"called __init__ with static={static}")
        print(f"calling the static method results in: {static()}")
        self.static = static

    def __get__(self, cls, instance):
        print(f"called __get__ with cls={cls} and instance={instance}")
        if cls is None:
            print(f"cls is None, calling self.instance.__get__(self)")
            return self.instance.__get__(self)
        else:
            print(f"cls is not None, returning self.static")
            return self.static
        # so this returns a method with no args in any case

    def instance(self, instance):
        print(f"called instance with instance={instance}")
        self.instance = instance
        return self


class MyClass:
    @StaticOrInstanceDescriptor
    def foo():
        return "you've called the static method foo on MyClass"

    @foo.instance
    def foo(self):
        return f"you've called the method foo on an instance of MyClass: {self}"


obj = MyClass()
print(obj.foo())
print(MyClass.foo())
