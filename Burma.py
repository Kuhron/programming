# parser for a programming language workspace with the working title of "Burma"

import os
import sys

burma_verb_list = ["equal","true","false"]

verb_dict = {
    "t":"Burma.true",
    "f":"Burma.false"
}

noun_dict = {
}

# bool_dict = {
# 	"t":"True",
# 	"f":"False"
# }

type_dict = {
	"\"":"str",
	"#":"float",
	# "?":"bool",
	"$":"var"
}

pos_to_suffixes_dict = {
	"Verb":["imp","hyp","cond"],
	"Noun":["nom","acc"]
}

imports = ["Burma","sys","os"]
import_lines = "from __future__ import print_function\n\n" + "\n".join([("import " + x) for x in imports]) + "\n\n"

nl = "\n"

def t(word):
	if word is None:
		return ""
	root = word.root

	if type(word) is Verb:
		if root in verb_dict:
			root = verb_dict[root]
		elif root in burma_verb_list:
			root = "Burma." + root
		else:
			root = root

		if word.mood == "imp":
		    return root
		elif word.mood == "hyp":
			return "if " + root
		elif word.mood == "cond":
			return "    " + root

	elif type(word) is Noun:
		python_root = noun_dict[root] if root in noun_dict else root
		if word.type == "str":
			return "\"" + python_root + "\""
		elif word.type == "bool":
			return bool_dict[word.root]
		elif word.type == "var":
			return word.root
		else:
			return python_root

	else:
		return ""

def burma_to_python_line(line):
	line = line.split()
	line = [i.split("-") for i in line]
	line = [i for i in filter(lambda x: x != "", line)]
	if line == []:
		return ""

	verb = get_only_burma_word_with_pos(line, "Verb")
	verb = Verb(verb) if verb else None
	direct_object = get_only_burma_word_with_suffix(line, "acc")
	direct_object = Noun(direct_object) if direct_object else None
	subject = get_only_burma_word_with_suffix(line, "nom")
	subject = Noun(subject) if subject else None

	if verb is None:
		return ""

	if verb.root == "assign":
		return ("    " if verb.mood == "cond" else "") + t(direct_object) + " = " + t(subject)

	result = ""
	result += t(verb) + "(" + \
		(t(subject) + ", " if subject else "") + \
		t(direct_object) + ")" + \
        (":" if verb.mood == "hyp" else "")

	return result

def get_burma_words_with_suffix(line, suffix):
	return [i for i in filter(lambda x: x[-1] == suffix, line)]

def get_only_burma_word_with_suffix(line, suffix):
	words = get_burma_words_with_suffix(line, suffix)
	if len(words) == 0:
		return None
	elif len(words) > 1:
		raise IndexError("More than one word found with suffix \"{0}\" in line \"{1}\"".format(suffix, line))
	else:
		return words[0]

def get_burma_words_with_pos(line, pos):
	result = []
	for suffix in pos_to_suffixes_dict[pos]:
		result += get_burma_words_with_suffix(line, suffix)
	return result

def get_only_burma_word_with_pos(line, pos):
	words = get_burma_words_with_pos(line, pos)
	if len(words) == 0:
		return None
	elif len(words) > 1:
		raise IndexError("More than one word found with part of speech \"{0}\" in line \"{1}\"".format(pos, line))
	else:
		return words[0]

def burma_to_python_file(filename):
	f = open(filename,"r")
	result = "\n".join([burma_to_python_line(line) for line in f.readlines()])
	f.close()

	while "\n\n" in result:
		result = result.replace("\n\n","\n")
	
	return result

class Verb:
	def __init__(self,verb):
		self.verb = verb
		self.root = verb[0]
		self.mood = verb[1]

class Noun:
	def __init__(self,noun):
		self.noun = noun
		self.type_char = noun[0][0]
		self.type = type_dict[self.type_char]
		self.root = noun[0][1:]
		self.case = noun[1]

def equal(a,b):
	return a == b

def true():
	return True

def false():
	return False

if __name__ == "__main__":
	burma_file = sys.argv[1]
	python_file = import_lines + burma_to_python_file(burma_file)
	f = open("Burma_output.py","w")
	f.write(python_file)
	f.close()
	# print("\n[resulting python code]\n{0}\n[end]".format(python_file))
	# print("[evaluating]\n")
	# eval(python_file)
	os.system("python Burma_output.py")
