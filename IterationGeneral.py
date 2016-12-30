def p(n):
	return n+n

def m(n):
	return n*n

def e(n):
	return n**n

def iterate(n,function_name="e"):
	k = function_name.count(".")
	function_name = function_name.replace(".","(")
	print("n =",n)
	s = (function_name+"(")*n + "n" + ")"*n*(k+1)
	print(s)
	return eval(s)

def i_1(n):
	return iterate(n,"iterate")

def i_2(n):
	s = ".".join(["i_1.iterate"]*n)
	return iterate(n,s)

def i_general(order):
	if order == 1:
		return i_1
	elif order == 2:
		return i_2
	elif order > 2:
		return lambda n: iterate(n,".".join(["i_general("+str(order-1)+").iterate"]*n))

def i_final(n):
	return i_general(n)(n)

def ii_1(n):
	return iterate(n,"i_final")

def ii_general(order):
	if order == 1:
		return ii_1
	elif order > 1:
		return lambda n: iterate(n,".".join(["ii_general("+str(order-1)+").iterate"]*n))

def ii_final(n):
	return ii_general(n)(n)

def i_k_1(k):
	if k == 1:
		return i_1
	elif k == 2:
		return ii_1
	elif k > 2:
		return lambda n: iterate(n,"i_k_final("+str(k-1)+")")

def i_k_general(k):
	return lambda order: (lambda n: iterate(n,".".join(["i_k_general("+str(k)+")("+str(order-1)+").iterate"]*n))) if order > 1 else i_k_1(k)

def i_k_final(k):
	return lambda n: i_k_general(k)(n)(n)

def i_all_final(n):
	return i_k_final(n)(n)

def jterate(n):
	return iterate(n,"i_all_final")

def kterate_general(k):
	if k == 1:
		return iterate
	elif k == 2:
		return jterate
	elif k > 2:
		return lambda n: iterate(n, "kterate_general("+str(k-1)+")")

def nterate(n):
	return kterate_general(n)(n)

# calling this stuff on 1 should give 1
#                       2             MemoryError
#                       larger        RAM fill

print(nterate(1))








