import random
# import re

PRIMES = [
    "PERSON",
    "GOOD",
    "NOT",
    "EQUAL",
    "EXIST",
    "THING",
    "BIG",
    "VERY",
    "MALE",
    "TIME",
    "I",
    "YOU",
    "BELIEVE"
]

WORDS = {
    key.lower() : key for key in PRIMES
}

def reverse_dictionary(d):
    result = {}
    for key in d:
        if d[key] in result:
            raise ValueError("Dictionary is not one-to-one.")
        else:
            result[d[key]] = key
    return result

REVERSE_WORDS = reverse_dictionary(WORDS)

def add_word(word,meaning):
    # m = []
    # for w in meaning.split():
    #   current_breakdown = w
    #   while set(current_breakdown.split()) - set(PRIMES) != set(): # while there are words in the breakdown that are not primes
    #       current_breakdown = " ".join([WORDS[ww] if ww not in PRIMES else ww for ww in current_breakdown.split()])
    #   m += [i for i in current_breakdown.split()]
    ml = to_metalanguage(meaning)
    WORDS[word] = ml
    REVERSE_WORDS[ml] = word


def show_missing_words(message):
    print(missing_words(message))

def to_metalanguage(expression):
    m = []
    
    for w in expression.split():
        current_breakdown = w
        while set(current_breakdown.split()) - set(PRIMES) != set(): # while there are words in the breakdown that are not primes
            for missing_word in missing_words(current_breakdown):
                get_meaning(missing_word)
            current_breakdown = " ".join([WORDS[ww] if ww not in PRIMES else ww for ww in current_breakdown.split()])
        m += [i for i in current_breakdown.split()]
    
    # remove double negations
    mm = []
    i = 0
    while i < len(m):
        if m[i] == "NOT" and m[i+1] == "NOT":
            # skip them both, don't add anything to mm
            i += 2
        else:
            mm.append(m[i])
            i += 1
    
    return " ".join(mm)

def semantically_equivalent(expression1, expression2):
    # very naive way for now, just see if the string expansion of the primes is the same
    # really should include syntax
    ml1 = to_metalanguage(expression1)
    ml2 = to_metalanguage(expression2)
    a = (ml1 == ml2)
    #if not a:
        #print(ml1)
        #print(ml2)
    return a

def semantically_subset(expression1,expression2):
    """
    Tells whether 1 is a semantic subset of 2. True if the former is a MORE SPECIFIC version of the latter.
    (NOT BIG PERSON, PERSON) returns True.
    (PERSON, NOT BIG PERSON) is not necessarily true, so returns False.
    """
    ml1 = to_metalanguage(expression1).split()
    ml2 = to_metalanguage(expression2).split()

    # print(ml1)
    # print(ml2)

    # for now just see if the first one is equal to the second one but with some words removed
    # but cannot ignore "NOT"
    # however, immediate false on hitting an unmatched "NOT" causes scope problems, as with "dwarf is person" throwing false due to "NOT BIG" # fixed
    i1 = 0
    for i2 in range(len(ml2)):
        while i1 < len(ml1) and ml1[i1] != ml2[i2]:
            # print("{0},{1}".format(ml1[i1],ml2[i2]))
            if ml1[i1] == "NOT" and ml1[i1+1] == ml2[i2]: # when the "NOT" negates the next thing in ml1, there is a contradiction between the expressions
                # print("input {0} is false by negation of element {1}".format("\""+expression1+" is "+expression2+"\"", ml2[i2]))
                return False
            i1 += 1
        if i1 >= len(ml1):
            # print("input {0} is false because some semantic components differ".format("\""+expression1+" is "+expression2+"\""))
            return False
        i1 += 1 # I had forgotten to advance BOTH i1 (done here) AND i2 (done by the for loop) when the elements matched
    return True

def missing_words(message):
    return set(message.split()) - set(WORDS)

def understood(message):
    return missing_words(message) == set()

def get_meaning(word):
    if word in WORDS:
        # do nothing
        return
    
    meaning = input("What does {0} mean? ".format(word))
    add_word(word, meaning)

def ask_for_expression():
    clarify(input("What would you like to tell me? "))

def clarify(message):
    for word in message.split():
        get_meaning(word)

def ask_for_question():
    question = input("What would you like to ask me? Please either ask 'what is X' or give an expression stating that 'X is Y'.\n")

    if question[:8] == "what is ": # user wants definition
        thing_to_define = question[8:]
        print("{0} is {1}.".format(thing_to_define,to_metalanguage(thing_to_define).lower()))

    elif " is " in question: # user asking if two things are the same
        if question[:5] == "does ": # redundant yes/no question particle
            question = question[5:]
        question = question.split(" is ")
        clarify(question[0])
        clarify(question[1])
        if semantically_equivalent(question[0],question[1]):
            print("That is correct by definition.")
        elif semantically_subset(question[0],question[1]): # e.g. "dwarf is small" should be True, where "dwarf" is more specific
            print("That is correct by subsetting.")
        else:
            print("That is not correct.")

    elif question[:5] == "does ": # user asking yes/no question, e.g. "does dog have eye"
        q = question[5:]
        print("I am not yet ready to process yes-or-no questions not using \"is\". Please write my algorithms already.")
        val = evaluate_truth(q)
        if val == None: # function evaluate_truth() still not returning things
            pass
        elif val:
            print("Yes!")
        else:
            print("No.")

    else:
        print("I don't understand your question.")

def evaluate_truth(expression): # example
    pass

add_word("bad","not good")
add_word("goodness","good thing")
add_word("badness","bad thing")
add_word("small","not big")
add_word("giant","big person")
add_word("dwarf","small person")
add_word("is","equal")
add_word("female","not male")

# print(WORDS)

# print(semantically_equivalent("badness","not goodness")) # True; works
# print(semantically_equivalent("good dwarf","not bad not giant")) # True; works
# print(semantically_equivalent("not not not not","")) # True; works

# show_missing_words("bad person hat evil") # {"hat","evil"}; works

test_PSRs = {
    "S":["A","BC"],
    "BC":["B","C"],
    "A":["(B)","A"],
    "B":["(C)","B"],
    "C":["(A)","C"]
}

def expand_PSRs(PSRs):
    """
    Looks at set of PSRs and returns a dictionary mapping each acceptable singlet and pair to a given type.
    Relies on a given sequence of types (of length 1 or 2) only being interpretable as one type. (trying to change this)
    """
    d = {}
    for rule_type in PSRs:
        #if rule_type == "S":
            #continue
        # don't add S into the collapsing unless to check at the very end of the collapses

        rule = PSRs[rule_type]
        mandatory_structure = tuple([i for i in filter(lambda x: "(" not in x, rule)]) # tuple
        
        optional_structure = tuple([i for i in map(lambda x: x[1:-1] if "(" in x else x, rule)]) # tuple
        # remove parentheses but allow that name of type has more than one character
        
        if mandatory_structure not in d:
            d[tuple(mandatory_structure)] = [rule_type]
        else:
            d[tuple(mandatory_structure)].append(rule_type)

        if mandatory_structure == optional_structure:
            # don't add stuff twice
            continue
        
        if optional_structure not in d:
            d[tuple(optional_structure)] = [rule_type]
        else:
            d[tuple(optional_structure)].append(rule_type)

    # print("result of expanding PSRs: ", d)
    return d

# unfortunately this (function PSR_check) didn't get fixed easily enough, so I rewrote the algorithm

# def PSR_check(input_types, output_type, PSRs): # even though this function and the helper contain a bunch of redundant stuff, don't mess with them for now
#     print(input_types)

#     rules = expand_PSRs(PSRs)

#     # # base case
#     # if len(input_types) == 1 and rules[tuple(input_types)] == output_type: # assume X is always an XP
#     #     return True

#     # if we hit a correct sentence-making binary pair:
#     # for future reference, this part should be OMITTED in favor of more flexible ways of determining whether the length-2 input returns the correct type
#     # i believe the below block accomplishes this
#     # try:
#     #     if output_type == "S" and "S" in rules[tuple(input_types)]:
#     #         return True
#     # except KeyError:
#     #     pass

#     # this block is more flexible than the previous one and so should be used instead of it
#     try:
#         if len(input_types) <= 2 and output_type in rules[tuple(input_types)]:
#             return True
#     except KeyError: # input types not in rules
#         pass

#     for i in range(len(input_types)-1):
#         if PSR_check_helper(input_types, output_type, PSRs, rules, i):
#             return True
#         # else, do nothing
#     return False


# def PSR_check_helper(input_types, output_type, PSRs, expanded_PSRs, i):
#     print(input_types)

#     # base case
#     if len(input_types) == 1 and input_types[0] == output_type: # assume X is always an XP
#         return True

#     # mandatory_structure = [i for i in filter(lambda x: "(" not in x, PSRs[output_type])]
#     # print(mandatory_structure)

#     # mandatory_structure_indices = [] # store the indices of the first occurrence of each part of the mandatory structure
#     # current_index = 0
#     # for i in mandatory_structure:
#     #     try:
#     #         mandatory_structure_indices.append(input_types[current_index:].index(i))
#     #     except ValueError: # item not found at or after current index, so mandatory structure is violated
#     #         return False

#     # finding out whether something is subsequence; do this to ensure the mandatory structure is found
#     # http://codegolf.stackexchange.com/questions/5529/
#     # s=lambda x,y:re.search('.*'.join(x),y) # needs to be converted to lists, or make lists into strings and then convert back

#     rules = expanded_PSRs

#     # if we hit a correct sentence-making binary pair:
#     try:
#         if len(input_types) == 2 and output_type in rules[tuple(input_types)]:
#             return True
#     except KeyError: # input types not in rules
#         pass

#     # used to be here: for i in range(len(input_types)-1):
#     # plus indent
#     # seq1 = tuple(input_types[i])
#     seq2 = tuple(input_types[i:i+2])
#     print("Trying sequence   {0} against rules.".format(seq2))

#     # if seq1 in rules: # this might create an infinite loop due to X mapping to XP always
#     #     return PSR_check(...)
#     if seq2 in rules:
#         print("this sequence was in the rules, with value",rules[seq2])
#         for t in rules[seq2]:
#             print(t)
#             if t == "S": # don't collapse S into a longer sequence
#                 continue
#             #t = rules[seq2]
#             print("Success. sequence {0} gives type {1}.".format(seq2,t))
            
#             collapsed_input_types = input_types[:i] + [t] + input_types[i+2:]
#             # out of bounds exception should not happen; [1,2,3][400:405] gives []
        
#             print("checking recursively on input",collapsed_input_types)
#             if PSR_check(collapsed_input_types,output_type,PSRs):
#                 # don't return False if we aren't done looking at the possible collapsed types for this sequence;
#                 # returning False is handled in the highest-level call of PSR_check if it never gets a True from any of the recursions
#                 return True
#             else:
#                 continue
#     else:
#         # single parts of speech may need to be converted to higher levels before they match the output type
#         for word_index in range(len(seq2)):
#             word = seq2[word_index]
#             print("seq2[word_index] is",word)
#             # if tuple([word]) not in rules:
#             #     continue
#             for t in rules[tuple([word])]: # note that tuple("V1") == ("V","1") because strings are iterable, so use list
#                 print(t)
#                 if t == seq2[word_index]:
#                     # prevent repeated trying with no changes
#                     continue
#                 collapsed_input_types = input_types[:word_index] + [t] + input_types[word_index+1:]
#                 if PSR_check(collapsed_input_types,output_type,PSRs):
#                     return True
#                 else:
#                     continue

#     # if no combination of binary collapses yields True, do not return anything from the helper! That way the next index can be tried.


class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self,item):
        self.queue.append(item)

    def dequeue(self):
        result = self.queue[0]
        self.queue = self.queue[1:]
        return result

    def is_empty(self):
        return self.queue == []


def get_compressions(compressing_rules,input_types):
    result = [[]]

    for i in range(len(input_types)):
        # get compressions for this index
        # all additions will propagate from the back
        old_result = result
        result = []
        for j in old_result:
            # print("j",j)
            j = j + [input_types[i]]
            Q = Queue()
            Q.enqueue(j)
            while not Q.is_empty():
                q = Q.dequeue()
                q1 = tuple([k for k in q[-1:]]) # this is just (q[-1],)
                q2 = tuple([k for k in q[-2:]]) if len(q) >= 2 else None
                if q2 in compressing_rules:
                    # print("compressing2", q2)
                    for compression in compressing_rules[q2]: # the compressions will just be one element, so have to put them in a tuple
                        Q.enqueue(q[:-2] + [compression])
                if q1 in compressing_rules:
                    # print("compressing1", q1)
                    for compression in compressing_rules[q1]:
                        if (compression,) != q1: # don't loop infinitely if the rule just turns X back into X itself
                            Q.enqueue(q[:-1] + [compression])
                result.append(q) # add each compression, including the ones where compression does not actually occur

    return result

def recognize(compressing_rules,input_types,output_type):
    return output_type in get_compressions(compressing_rules,input_types)

# print(get_compressions(expand_PSRs(test_PSRs),["A","C","B","C"])) # works! just read them and double check if you doubt

def PSR_check(arg1, arg2, arg3): # to keep old syntax for tests
    return recognize(expand_PSRs(arg3),arg1,[arg2])

# print(PSR_check(["A"],"A",test_PSRs)) # True
# print(PSR_check(["A","BC"],"S",test_PSRs)) # True
# print(PSR_check(["A","C"],"C",test_PSRs)) # True
# print(PSR_check(["A","B"],"B",test_PSRs)) # False
# print(PSR_check(["C","B","A"],"A",test_PSRs)) # True
# print(PSR_check(["A","C","B","C"],"S",test_PSRs)) # True
# print(PSR_check(["B","A","A","C","B","A","C"],"S",test_PSRs)) # True
# print(PSR_check(["A","C","B","A","C"],"BC",test_PSRs)) # True
# print(PSR_check(["A","C","B","A","C","B"],"BC",test_PSRs)) # False
# print(PSR_check(["B","A","C","B","C","B","A"],"S",test_PSRs)) # False
# all of these work!


english_PSRs = {
    # english psrs so far:
    # keep BINARY for ease of parsing
    "A" : ["(Adv)","A"],
    "Adv" : ["(Adv)","Adv"],
    "VP" : ["(Adv)","V1"],
    "V1" : ["V","(N)"],
    "N" : ["(A)","N"],
    "S" : ["N","VP"]
}

def word_types(s, word_dict):
    word_list = s.split()
    return[word_dict[i] for i in word_list]

english_word_dict = {
    "man":"N",
    "woman":"N",
    "see":"V",
    "big":"A",
    "easily":"Adv",
    "tall":"A",
    "very":"Adv"
}

# print(PSR_check(word_types("big man",english_word_dict),"N",english_PSRs)) # True
# print(PSR_check(word_types("man see woman",english_word_dict),"S",english_PSRs)) # True
# print(PSR_check(word_types("man see very woman",english_word_dict),"S",english_PSRs)) # False
# print(PSR_check(word_types("very easily very very big tall woman",english_word_dict),"N",english_PSRs)) # True
# print(PSR_check(word_types("very very very tall man",english_word_dict),"A",english_PSRs)) # False
# print(PSR_check(word_types("tall man easily see",english_word_dict),"S",english_PSRs)) # True
# print(PSR_check(word_types("see woman",english_word_dict),"S",english_PSRs)) # False
# print(PSR_check(word_types("easily see",english_word_dict),"VP",english_PSRs)) # True
# these all work!

def result_length(lst): # auxiliary to compress(); counts elements covered in a working compression result, allows for multiple uses of subsequences
    return sum([len(lst[i]) for i in range(len(lst))])

# def compress(sequence, subsequences):
#     # sequence = tuple(sequence)
#     # subsequences = tuple([tuple(i) for i in subsequences])

#     # for i in range(len(sequence)):
#     #     sequence_i = sequence[:i+1]

#     #     # cut off all the subsequences at the ith sequence element
#     #     #subsequences_i = set((sub[:min(sub.index(sequence[i]),len(sub))] if sequence[i] in sub else () for sub in subsequences))
#     #     subsequences_i = set()
#     #     for sub in subsequences:
#     #         if i < sequence.index(sub[0]):
#     #             # this sequence hasn't started yet, so there is nothing to add
#     #             continue
#     #         else:
#     #              pass # idk

#     #     print(subsequences_i)

#     if len(sequence) == 1 and [sequence[0]] in subsequences: # same as "and sequence in subsequences", but maybe clearer this way
#         result = [[sequence[0]]]
#         #print("result from sequence of length 1:",result)
#         return result

#     memoized_results = [[] for i in range(len(sequence))] # memoize old results so we don't have to make recursive calls; index is len(s)-1 = i

#     result = []
#     for i in range(len(sequence)):
#         # build the result one index at a time, only replacing earlier partitions with larger ones whose ends are found at index i
#         s = sequence[:i+1]
#         subs = [list(sss) for sss in list({tuple(sub[:i+1]) for sub in subsequences})]
#         #print("s:",s)

#         current_n_subsequences = len(result)

#         enders = [k for k in subs] # initialize possible ending subsequences, to be narrowed down to the longest that doesn't make the result worse
#         best_ender = None # for now, take the longest ender that does not break a sequence apart (this is still suboptimal in some cases, though)
#         recursive_result = None # initialize recursive result for adding to best ender found
#         best_recursive_result = None
#         # e.g. compressing [1,2,3,4,5,6,7] with [[1],[1,2,3,4],[2,3,4,5,6,7]] will refuse to break [1,2,3,4] and actually fails to find any solution
        
#         for j in range(i+1):
#             enders = [k for k in filter(lambda x: len(x) == j+1 and x[-j-1:] == s[-j-1:], subs)] # this must be unique or nonexistent
#             if len(enders) > 1:
#                 raise ValueError("Ender for i={0}, j={1} is not unique.".format(i,j))
#             print("enders for i={2}, j={0}: {1}".format(j,enders,i))

#             if len(enders) == 1:
#                 ender = enders[0]
#             else: # no enders of this length
#                 continue
#             print("ender:",ender) # 360 no scope

#             # j+1 is len(ender), i+1 is len(s)
#             # print("making recursive call")
#             # recursive_result = compress(s[:i+1-(j+1)],subsequences) # actually recursive; no need to do this
#             recursive_result = memoized_results[i+1-(j+1)-1] # name is misleading now, but whatever
#             #print("retrieved memoized result (for i={0}, j={1}):".format(i,j), recursive_result)
            
#             if (
#                 len(result) != 0 and # fine: if there is no result, we must use the ender
#                 len(ender) != len(s) - result_length(result) and # fine: if the ender fills in all the space after the result, we must use it
#                 (
#                     len(s) != len(sequence) or # normally an ender that breaks up previous compressions too much (makes too many subsequences necessary
#                         # before it) is not considered for use, but at the end we will be forced to use one
#                     ((result_length(result) + len(best_ender)) if best_ender != None else -1) == len(s) # the best ender does not fill in the remaining space
#                     # ^^^ NEGATING THIS WILL FAIL because you are saying -1 != len(s) (ALWAYS TRUE) if there is no best ender
#                     # note that if we are at the very end of the input sequence and there is still space to be filled, we should take an ender that fills it
#                     # at the expense of previous solutions which could not cover the whole sequence
#                 ) and
#                 len(recursive_result) >= current_n_subsequences # the result forced by using this ender is worse in terms of number of subsequences used
#             ): # dear god, these booleans

#                 # if it's equal, still worse because we have to add one for the ender
#                 # cases excluded from this block:
#                 #     - result is empty; we must add any ender available
#                 #     - the ender does not overlap the previous result; we must add it or else we will not cover the whole sequence
#                 print("ender {0} made things worse because".format(ender), 
#                     (("\ns is not the entire sequence;" if len(s) != len(sequence) else "") +
#                         ("\nthe ender does not fill in the rest of s;" if 
#                             (result_length(result) + (len(best_ender)) if best_ender != None else -1) == len(s) else "")))
#                 continue
#             # elif ( # !!!!! MOVING THESE CONDITIONS HERE BROKE THE CODE, BUT HOPEFULLY IT WILL BE EASIER TO UNDERSTAND IN THIS ORGANIZATION
#             #     len(s) == len(sequence) and # normally an ender that breaks up previous compressions too much (makes too many subsequences necessary
#             #     # before it) is not considered for use, but at the end we will be forced to use one
#             #     (best_ender != None and (result_length(result) + len(best_ender)) != len(s))
#             # ): # the best ender does not fill in the remaining space
#                 # ^^^ NEGATING THIS WILL FAIL because you are saying -1 != len(s) (ALWAYS TRUE) if there is no best ender
#                 # note that if we are at the very end of the input sequence and there is still space to be filled, we should take an ender that fills it
#                 # at the expense of previous solutions which could not cover the whole sequence
#                 print("forced to use ender {0} because we must cover the input sequence".format(ender))
#                 best_ender = ender
#                 best_recursive_result = recursive_result # is this really correct to do in this situation? make sure we are getting the CORRECT memoized result

#             else: # this ender makes things better
#                 print("ender {0} made things better".format(ender))
#                 best_ender = ender
#                 best_recursive_result = recursive_result

#         if best_ender != None and best_recursive_result != None:
#             result = best_recursive_result + [best_ender]
#         print("result for i={0}:".format(i),result)
#         memoized_results[i] = result

#     print("result:",result)

#     if result_length(result) != len(sequence):
#         # something went wrong; the whole sequence was not covered (just return None) or there is a bug (the result has been printed if debugging)
#         #print("returning None")
#         return None
#     else:
#         return result

def compress_new(sequence, subsequences):
    # for sequences longer than 1, start by memoizing all possibilities for sequence[:i] for all i < len(sequence)
    memo = {0:None}
    for i in range(1,len(sequence)):
        helper_result = compress_new_helper(sequence[:i],subsequences,memo)
        memo[i] = helper_result # possibilities only, not a choice

    possibilities = compress_new_helper(sequence,subsequences,memo)
    # print("possibilities:",possibilities)
    min_len = min([len(i) for i in possibilities])
    shortest_possibilities = [i for i in filter(lambda x: len(x) == min_len, possibilities)]
    return random.choice(shortest_possibilities)

def compress_new_helper(sequence,subsequences,memo):
    if len(sequence) == 1:
        if sequence in subsequences:
            return [[sequence]]
        else:
            return None

    if len(sequence) in memo:
        #print("memoized result for length {0}:".format(len(sequence)),memo[len(sequence)])
        return memo[len(sequence)]

    possibilities = []
    enders = [i for i in filter(lambda x: sequence[-len(x):] == x, subsequences)]
    # print("enders:",enders)
    for ender in enders:
        if len(sequence) - len(ender) in memo:
            #print("memoized sub-result for length {0}:".format(len(sequence)-len(ender)),memo[len(sequence)-len(ender)])
            a = memo[len(sequence)-len(ender)]
        else:
            try:
                print("making recursive call for length {0}; this should not happen from the top down".format(len(sequence)-len(ender)))
                # turns out this was only happening for length 0 due to it not being found in memo, but I have now fixed that by adding 0:None
                a = compress_new_helper(sequence[:-len(ender)],subsequences,memo)
            except RuntimeError:
                print("Segmentation fault (core dumped)")
                import sys
                sys.exit()
        if ender == sequence:
            possibilities.append([ender])
            continue
        if a != None:
            for aa in a:
                # print("aa should be a list of subsequences:",aa)
                # each ender is a subsequence, form [1,2,3]
                # a is a list of possibilities (each of which is a list of subsequences), form [[[1],[2]],[[1,2]]]
                # aa is a possibility, form [[1],[2]]
                # print("adding possibility:",aa + [ender])
                possibilities.append(aa + [ender])

    # print("possibilities should be a list of lists of subsequences:",possibilities)
    return possibilities

def compress(x,y): # keep old syntax
    return compress_new(x,y)

# print(compress([1,2,3],[[1],[2],[3],[1,2,3],[3,4,5,6,7],[4,5,6,7]])) # [[1,2,3]]
# print(compress([1,2,3,4,5,6,7],[[1],[2],[3],[1,2,3],[3,4,5,6,7],[4,5,6,7]])) # [[1,2,3],[4,5,6,7]]
# print(compress([1,2,1,3,1,1,2,1],[[1],[1,2],[3,1],[2,1,2],[7]])) # [[1,2],[1],[3,1],[1,2],[1]]; should simply omit impossible subsequences rather than error
# print(compress([1,2,3,4,5,6,7],[[1],[1,2,3,4],[2,3,4,5,6,7]])) # [[1],[2,3,4,5,6,7]]
# print(compress([1],[[1],[2]])) # [[1]]
# print(compress([1,2,1],[[1],[2],[1,2]])) # [[1,2],[1]]
# print(compress([1,2,1,2,1],[[1],[1,2],[2,1]])) # multiple solutions: 1. [[1],[2,1],[2,1]], 2. [[1,2],[1,2],[1]] # either is fine, but it must pick one
# print(compress([1,2,1,2,1],[[1],[1,2,1],[2,1,2]])) # [[1],[2,1,2],[1]]
# print(compress([random.choice(range(10)) for i in range(50)],[[i]*j for i in range(10) for j in range(50)])) # all contiguous groupings of digits


def from_metalanguage(expression):
    e = to_metalanguage(expression).split() # ensure we are actually converting from metalanguage
    w = [WORDS[key].split() for key in WORDS]
    # print(w)
    compression = compress(e,w)
    return " ".join([REVERSE_WORDS[" ".join(subsequence)] for subsequence in compression])

# print(from_metalanguage("small person is not male")) # "dwarf is female"
# print(from_metalanguage("dwarf is not male")) # bad input; should have been in metalanguage, but should still give "dwarf is female"
# these work

def converse():
    try:
        while True:
            a = input("1. tell me something; 2. ask me something ")
            if a == "1":
                ask_for_expression()
            elif a == "2":
                ask_for_question()
            else:
                print("Invalid input: {0}".format(a))

    except KeyboardInterrupt:
        print(WORDS)

# these tests must pass before further progress can be made
# print(semantically_subset("dwarf","person")) # True
# print(semantically_subset("person","dwarf")) # not necessarily -> False
# print(semantically_subset("small","dwarf")) # not necessarily -> False
# print(semantically_subset("dwarf","small")) # True
# all work now

converse()
















