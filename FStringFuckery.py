Python 3.8.9 (default, Apr  3 2021, 01:02:10) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> f"{degree sign}"
  File "<fstring>", line 1
    (degree sign)
            ^
SyntaxError: invalid syntax
>>> a = 4
>>> b = a
>>> f"{b=}"
'b=4'
>>> f"\N"
  File "<stdin>", line 1
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 0-1: malformed \N character escape
>>> f"{4 if False else 5}"
'5'
>>> def
KeyboardInterrupt
>>> f"{(lambda: 5)()}
  File "<stdin>", line 1
    f"{(lambda: 5)()}
                    ^
SyntaxError: EOL while scanning string literal
>>> f"{(lambda: 5)()}"
'5'
>>> f"{(lambda f: f(f))(lambda f: f(f))}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in <lambda>
  File "<stdin>", line 1, in <lambda>
  File "<stdin>", line 1, in <lambda>
  [Previous line repeated 996 more times]
RecursionError: maximum recursion depth exceeded
>>> 
>>> f"{}".format(3)
  File "<stdin>", line 1
SyntaxError: f-string: empty expression not allowed
>>> "{}".format(3)
'3'
>>> a
4
>>> F"a"
'a'
>>> f"a"
'a'
>>> F"{a}"
'4'
>>> a = "sdf"
>>> F"{a}"
'sdf'
>>> fR"{a}"
'sdf'
>>> s = f"
KeyboardInterrupt
>>> s = f"{x}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
>>> s = "{x}"
>>> s.format(x=4)
'4'
>>> f"{x=4; x*x}"
  File "<stdin>", line 1
SyntaxError: f-string: expecting '}'
>>> x
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
>>> a
'sdf'
>>> f"{a      =       }"
"a      =       'sdf'"
>>> f"{234:b}
  File "<stdin>", line 1
    f"{234:b}
            ^
SyntaxError: EOL while scanning string literal
>>> f"{234:b}"
'11101010'
>>> f"{234:o}"
'352'
>>> f"{234:d}"
'234'
>>> f"{234:x}"
'ea'
>>> f"{234:X}"
'EA'
>>> f"{0234:X}"
  File "<fstring>", line 1
    (0234)
        ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
>>> f"{0o234:X}"
'9C'
>>> 0234
  File "<stdin>", line 1
    0234
       ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
>>> 0o234
156
>>> 0x234
564
>>> 0b234
  File "<stdin>", line 1
    0b234
     ^
SyntaxError: invalid digit '2' in binary literal
>>> 0boeaducgthi
  File "<stdin>", line 1
    0boeaducgthi
     ^
SyntaxError: invalid binary literal
>>> 0b!!!
  File "<stdin>", line 1
    0b!!!
     ^
SyntaxError: invalid binary literal
>>> 0xa.uhciprch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'int' object has no attribute 'uhciprch'
>>> 0x,.p.hpith
  File "<stdin>", line 1
    0x,.p.hpith
     ^
SyntaxError: invalid hexadecimal literal
>>> 0xx
  File "<stdin>", line 1
    0xx
     ^
SyntaxError: invalid hexadecimal literal
>>> 0xd
13
>>> 0xg
  File "<stdin>", line 1
    0xg
     ^
SyntaxError: invalid hexadecimal literal
>>> f"{0xaaa:8d}"
'    2730'
>>> f"{0xaaa:8o}"
'    5252'
>>> f"{0xaaa:8b}"
'101010101010'
>>> f"{0b1101100111001:x}"
'1b39'
>>> f"{0b1101100111001:d}"
'6969'
>>> f"{0b1101100111001:}"
'6969'
>>> f"{0b1101100111001:o}"
'15471'
>>> f"{0b1101100111001}"
'6969'
>>> a = "x"
>>> f"{0b1101100111001:{a}}"
'1b39'
>>> f"{0b1101100111001:{n}{base}}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'n' is not defined
>>> n = 12
>>> base = 5
>>> f"{0b1101100111001:{n}{base}}"
'                                                                                                                         6969'
>>> f"{0b110110011100{base}:{n}{base}}"
  File "<fstring>", line 1
    (0b110110011100{base})
                   ^
SyntaxError: invalid syntax
>>> f"{0b110110011100:{n}{base}}"
'                                                                                                                         3484'
>>> None.__str__()
'None'
>>> None.__repr__()
'None'
>>> f = lambda x: x.__str__() == x.__repr__()
>>> f(3)
True
>>> f(int)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in <lambda>
TypeError: descriptor '__str__' of 'object' object needs an argument
>>> f(type)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in <lambda>
TypeError: descriptor '__str__' of 'object' object needs an argument
>>> f(type)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in <lambda>
TypeError: descriptor '__str__' of 'object' object needs an argument
>>> f(f)
True
>>> f(())
True
>>> f([])
True
>>> f((3,))
True
>>> f(Exception)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in <lambda>
TypeError: descriptor '__str__' of 'BaseException' object needs an argument
>>> f(Exception.__str__)
True
>>> f(locals)
True
>>> f(locals())
True
>>> f({})
True
>>> f(0.)
True
>>> f(0.)
True
>>> f(True)
True
>>> f("True")
False
>>> f("a")
False
>>> f('a')
False
>>> "a".__str__()
'a'
>>> "a".__repr__()
"'a'"
>>> f"{x:a}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
>>> x = 4
>>> f"{x:a}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Unknown format code 'a' for object of type 'int'
>>> f"{x!a}"
'4'
>>> f"{x!u}"
  File "<stdin>", line 1
SyntaxError: f-string: invalid conversion character: expected 's', 'r', or 'a'
>>> x = "ů́hu̩̯"
>>> f"{x!a}"
"'u\\u030a\\u0301hu\\u0329\\u032f'"
>>> x = 3.412983472
>>> f"{x:.{n}f}"
'3.412983472000'
>>> for n in range(10): print(f"{x:.{n}f}")
... 
3
3.4
3.41
3.413
3.4130
3.41298
3.412983
3.4129835
3.41298347
3.412983472
>>> for n in range(15): print(f"{x:.{n}f}")
... 
3
3.4
3.41
3.413
3.4130
3.41298
3.412983
3.4129835
3.41298347
3.412983472
3.4129834720
3.41298347200
3.412983472000
3.4129834720000
3.41298347200000
>>> for n in range(15): print(f"{x:.}")
KeyboardInterrupt
>>> f"{x:%}"
'341.298347%'
>>> f"{x:3%}"
'341.298347%'
>>> f"{x:.3%}"
'341.298%'
>>> f"{x:.2%}"
'341.30%'
>>> f"{x:.2f}"
'3.41'
>>> f"{x:.2%}"
'341.30%'
>>> f"{x:15}"
'    3.412983472'
>>> f"{'x':15}"
'x              '
>>> f"{5:15}"
'              5'
>>> f"{True:15}"
'              1'
>>> f"{True:15!s}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Invalid format specifier
>>> f"{True!s:15}"
'True           '
>>> f"{True!r:15}"
'True           '
>>> f"{True!a:15}"
'True           '
>>> f"{True:15}"
'              1'
>>> f"{{}:15}"
  File "<stdin>", line 1
SyntaxError: f-string: single '}' is not allowed
>>> f"{d:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'd' is not defined
>>> d = {}
>>> f"{d:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to dict.__format__
>>> f"{[]:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to list.__format__
>>> f"{():15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to tuple.__format__
>>> f"{(4,):15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to tuple.__format__
>>> f"{None:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to NoneType.__format__
>>> f"{type:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to type.__format__
>>> f"{type.__format__:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to method_descriptor.__format__
>>> f"{type.__format__.__format__:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to builtin_function_or_method.__format__
>>> f"{type.__format__.__format__.__format__:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to builtin_function_or_method.__format__
>>> f"{type.__format__.__format__.__format__:True}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to builtin_function_or_method.__format__
>>> f"{type.__format__.__format__.__format__:type}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to builtin_function_or_method.__format__
>>> f"{type.__format__.__format__.__format__:{}}"
  File "<stdin>", line 1
SyntaxError: f-string: empty expression not allowed
>>> f"{type.__format__.__format__.__format__:{None}}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to builtin_function_or_method.__format__
>>> try:f"{type.__format__.__format__.__format__:{None}}"
... except Exception as e:
...  print(e.__dict__)
... 
{}
>>> dir(Exception)
['__cause__', '__class__', '__context__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__suppress_context__', '__traceback__', 'args', 'with_traceback']
>>> sorted(dir(Exception))
['__cause__', '__class__', '__context__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__suppress_context__', '__traceback__', 'args', 'with_traceback']
>>> x
3.412983472
>>> f"{x:^15}"
'  3.412983472  '
>>> f"{x:>15}"
'    3.412983472'
>>> f"{x:<15}"
'3.412983472    '
>>> f"{x:^15}"
'  3.412983472  '
>>> f"{x:^25}"
'       3.412983472       '
>>> f"{x:k^25}"
'kkkkkkk3.412983472kkkkkkk'
>>> f"{x:^^25}"
'^^^^^^^3.412983472^^^^^^^'
>>> f"{x:^^^25}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Invalid format specifier
>>> f"{x:abc^25}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Invalid format specifier
>>> f"{x:{a}^25}"
'xxxxxxx3.412983472xxxxxxx'
>>> a
'x'
>>> a = "xyz"
>>> f"{x:{a}^25}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Invalid format specifier
>>> def has_length_one(s):
...  try:
...   f"{s:{s}^1}"
...  except:
...   return False
...  return Tre
... 
>>> Tre = True
>>> has_length_one("4")
True
>>> has_length_one("4u")
False
>>> has_length_one("u")
True
>>> has_length_one(SyntaxError)
False
>>> has_length_one([3])
False
>>> has_length_one([])
False
>>> has_length_one(type)
False
>>> has_length_one()
KeyboardInterrupt
>>> exec(f"exec(f\"print(\\\"\\n\\\\")\")\")
  File "<stdin>", line 1
    exec(f"exec(f\"print(\\\"\\n\\\\")\")\")
                                           ^
SyntaxError: unexpected character after line continuation character
>>> exec(f"exec(f\"print(\\\"\\n\\\")\")\")
  File "<stdin>", line 1
    exec(f"exec(f\"print(\\\"\\n\\\")\")\")
                                          ^
SyntaxError: EOL while scanning string literal
>>> exec(f"exec(f\"print(\\\"\\n\\\")\")\")
KeyboardInterrupt
>>> print("\n")


>>> print(\"\\n\")
  File "<stdin>", line 1
    print(\"\\n\")
                 ^
SyntaxError: unexpected character after line continuation character
>>> "print(\"\\n\")"
'print("\\n")'
>>> exec("print(\"\\n\")")


>>> "exec(\"print(\\\"\\\\n\\\")\")"
'exec("print(\\"\\\\n\\")")'
>>> exec("exec(\"print(\\\"\\\\n\\\")\")")


>>> exec(f"exec(\"pri{s}t(\\\"\\\\{s}\\\")\")")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 1, in <module>
  File "<string>", line 1
    pri{x}t("\{x}")
       ^
SyntaxError: invalid syntax
>>> s = "n"
>>> exec(f"exec(\"pri{s}t(\\\"\\\\{s}\\\")\")")


>>> exec(f"exec(\"pri{s}t(\\\"\{s:\^7}")\")")
  File "<stdin>", line 1
    exec(f"exec(\"pri{s}t(\\\"\{s:\^7}")\")")
                                            ^
SyntaxError: unexpected character after line continuation character
>>> exec(f"exec(\"pri{s}t(\\\"\{s:\\^7}")\")")
  File "<stdin>", line 1
    exec(f"exec(\"pri{s}t(\\\"\{s:\\^7}")\")")
                                             ^
SyntaxError: unexpected character after line continuation character
>>> exec(f"exec(\"pri{s}t(\\\"\{s:'\'^7}")\")")
  File "<stdin>", line 1
    exec(f"exec(\"pri{s}t(\\\"\{s:'\'^7}")\")")
                                              ^
SyntaxError: unexpected character after line continuation character
>>> exec(f"exec(\"pri{s}t(\\\"\{s:{bs}^7}")\")")
  File "<stdin>", line 1
    exec(f"exec(\"pri{s}t(\\\"\{s:{bs}^7}")\")")
                                               ^
SyntaxError: unexpected character after line continuation character
>>> bs = "\"
  File "<stdin>", line 1
    bs = "\"
           ^
SyntaxError: EOL while scanning string literal
>>> bs = "\="
>>> bs = "\\"
>>> bs = "\"
  File "<stdin>", line 1
    bs = "\"
           ^
SyntaxError: EOL while scanning string literal
>>> exec(f"exec(\"pri{s}t(\\\"\{s:{bs}^7}")\")")
  File "<stdin>", line 1
    exec(f"exec(\"pri{s}t(\\\"\{s:{bs}^7}")\")")
                                               ^
SyntaxError: unexpected character after line continuation character
>>> f"{s:{bs}^7}"
'\\\\\\n\\\\\\'
>>> f"{s:{bs}^1}"
'n'
>>> f"{s:{bs}^2}"
'n\\'
>>> f"{s:{bs}^3}"
'\\n\\'
>>> print(f"{s:{bs}^7}")
\\\n\\\
>>> s
'n'
>>> len(bs)
1
>>> exec(f"exec(\"pri{s}t(\\\"\\\\{s}\\\")\")")


>>> exec(f"exec(\"pri{s}t(\\\"\\{s:{bs}^2}\")\")")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 1, in <module>
  File "<string>", line 1
    print("
          ^
SyntaxError: EOL while scanning string literal
>>> exec(f"exec(\"pri{s}t(\\\"\\{s:{bs}^3}\")\")")


>>> q = "\""
>>> exec(f"exec(\"pri{s}t(\\\"\\{s:{bs}^3}{q})\")")


>>> exec(f"exec(\"pri{s}t(\\\"\\{s:{bs}^3}{q}){q})")


>>> exec(f"exec(\"pri{s}t({q:{bs}^3}{s:{bs}^3}{q}){q})")


>>> f"{s=}"
"s='n'"
>>> len(f"{s=}")
5
>>> len(f"{4=}")
3
>>> exec(f"exec(\"pri{s}t({q:{bs}^3}{s:{bs}^len(f"{4=}")}{q}){q})")
  File "<stdin>", line 1
    exec(f"exec(\"pri{s}t({q:{bs}^3}{s:{bs}^len(f"{4=}")}{q}){q})")
                                                  ^
SyntaxError: invalid syntax
>>> exec(f"exec(\"pri{s}t({q:{bs}^3}{s:{bs}^len(f\"{{4=}}\")}{q}){q})")
  File "<fstring>", line 1
    ({4=})
       ^
SyntaxError: invalid syntax
>>> exec(f"exec(\"pri{s}t({q:{bs}^3}{s:{bs}^{len(f\"{{4=}}\")}}{q}){q})")
  File "<stdin>", line 1
SyntaxError: f-string expression part cannot include a backslash
>>> exec(f"exec(\"pri{s}t({q:{bs}^3}{s:{bs}^{{len(f\"{{4=}}\")}}}{q}){q})")
  File "<stdin>", line 1
SyntaxError: f-string expression part cannot include a backslash
>>> 
KeyboardInterrupt
>>> len(f"{4=}")
3
>>> exec(f"exec(\"pri{s}t({q:{bs}^3}{s:{bs}^3}{q}){q})")


>>> exec(f"exec({q}pri{s}t({q:{bs}^3}{s:{bs}^3}{q}){q})")


>>> exec(f"exec({q}pri{s}t({q:{bs}^3}{s:{bs}^3}{q}){q}))
[1]+  Stopped                 python3.8
wesley@lumen:~/programming$ fg
python3.8
^[[A^[[A^[[A^C
KeyboardInterrupt
>>> exec(f"exec({q}pri{s}t({q:{bs}^3}{s:{bs}^3}{q}){q})")


>>> a = 4
>>> exec(f"{a=}")
>>> a
4
>>> exec(f"{a=}{a}")
>>> a
44
>>> f"{1234:,}"
'1,234'
>>> f"{12345678:,}"
'12,345,678'
>>> f"{12345678::}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Unknown format code ':' for object of type 'int'
>>> f"{12345678:;}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Unknown format code ';' for object of type 'int'
>>> f"{12345678:'}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Unknown format code ''' for object of type 'int'
>>> f"{12345678:4}"
'12345678'
>>> f"{12345678:i}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Unknown format code 'i' for object of type 'int'
>>> f"{12345678:,,}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Cannot specify both ',' and '_'.
>>> f"{12345678:_}"
'12_345_678'
>>> f"{12345678:_,}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Cannot specify both ',' and '_'.
>>> f"{12345678: }"
' 12345678'
>>> f"{12345678:     }"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Invalid format specifier
>>> f"{12345678:     ,}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Invalid format specifier
>>> f"{12345678:a}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Unknown format code 'a' for object of type 'int'
>>> f"{12345678:,a}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Cannot specify ',' with 'a'.
>>> f"{12345678:,4}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Cannot specify ',' with '4'.
>>> f"{12345678:,;}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Cannot specify ',' with ';'.
>>> f"{12345678:,c}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Cannot specify ',' with 'c'.
>>> f"{12345678:,'}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Cannot specify ',' with '''.
>>> f"{12345678:,.3f}"
'12,345,678.000'
>>> f"{True:15}")
  File "<stdin>", line 1
    f"{True:15}")
                ^
SyntaxError: unmatched ')'
>>> f"{True:15}"
'              1'
>>> f"{None:15}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: unsupported format string passed to NoneType.__format__
>>> f"{True:e}"
'1.000000e+00'
>>> f"{True:%Y}"
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Invalid format specifier
>>> from datetime import datetime
>>> now = datetime.utcnow()
>>> f"{now:%Y}"
'2022'
>>> f"{True:04}"
'0001'
>>> f"{True:+04}"
'+001'
>>> f"{True:-04}"
'0001'
>>> 
