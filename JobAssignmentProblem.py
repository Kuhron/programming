# Job Assignment Problem
# given n people and m jobs, and a "fit function" telling how fit each person is for each job,
# find the arrangement of people into jobs that maximizes total fit

import random

import math
from math import isnan as isNaN

NaN = float("nan")
Inf = float("inf")

def all_NaN(matrix):
    for row in matrix:
        for x in row:
            if not isNaN(x):
                # NaN is not equal to itself, so we have an actual number here
                return False
    return True

# print(all_NaN([[NaN,NaN],[NaN,NaN]])) # True # works
# print(all_NaN([[NaN,NaN],[NaN,4]])) # False # works

def transpose(matrix):
	n_row = len(matrix)
	n_col = len(matrix[0])
	return [[matrix[i][j] for i in range(n_row)] for j in range(n_col)]

# print(transpose([[1,2,3,4],[5,6,7,8]])) # works

def is_fully_assigned(matrix):
    """
    Takes an assignment matrix and determines whether all possible assignments have been made;
    i.e. returns False if there is a row i and a column j which are both all zero.
    """
    n_row = len(matrix)
    n_col = len(matrix[0])
    if n_row > n_col:
        matrix = transpose(matrix)
    n_row = len(matrix)
    n_col = len(matrix[0])
    
    return not any(all(matrix[i][j] == 0 for j in range(n_col)) for i in range(n_row)) # return: no row is empty

# print(is_fully_assigned([[0,0,0],[1,0,0],[0,1,0]])) # False
# print(is_fully_assigned([[0,0,1],[1,0,0],[0,1,0]])) # True
# print(is_fully_assigned([[0,0],[1,0],[0,1]])) # True
# print(is_fully_assigned([[0,0,0],[1,0,0]])) # False
# print(is_fully_assigned([[0,0,1],[1,0,0]])) # True
# all work


def jap(m):
    """
    Takes input m (the n*m matrix where element (i,j) tells how fit person i is for job j).
    Returns an n*m matrix where element (i,j) is 1 if person i is assigned to job j and 0 otherwise.
    """

    original_m = [[m[i][j] for j in range(len(m[0]))] for i in range(len(m))]
    # print("original matrix:",original_m)

    # Note that we assume WLOG n <= m, so we assign each person to one job.
    # If there are more people than jobs, then note the symmetry of the problem, so just transpose m, solve, and then transpose the result.
    need_to_transpose = len(m) > len(m[0])
    if need_to_transpose:
        m = transpose(m)

    N = len(m)
    M = len(m[0])

    pairs = [] # to be list of tuples (set of tuples could also work)

    #if all_NaN(m): # if it's somehow already all NaN
        #raise ValueError("Input matrix is all NaNs.")

    limiter = 0
    limit = 10**4
    while not all_NaN(m): # m will be all NaNs when we are done assigning jobs
        limiter += 1
        if limiter > limit:
            raise RuntimeError("While loop ran too many times.")

        # print("this remnant of original matrix:",m)

        for r in range(N): # for each row
            row_without_NaN = [i for i in filter(lambda x: not isNaN(x), m[r])]
            if row_without_NaN == []:
                # print("skipping normalizing row {0} due to all NaN".format(r))
                continue
            best = max(row_without_NaN) # best fit value for this person
            # print("best val:",best)
            m[r] = [i-best for i in m[r]] # subtract best value from everything in the row; everything will end up non-positive
            # print("normalized row:",m[r])

        # print("THIS SHOULD NOT BE SKIPPED IN ANY CALL")

        # method: find the rows with the greatest gap between the best and the second-best columns; these must be given highest priority
        # keep in mind that a row which is all NaN will have been skipped already, so if second_bests is all NaN then there is one zero in the row
        # if there is just one zero in one row, then that pair should be part of the solution! this is most often the form of the base case
        best_pair = None
        best_pair_found = False
        second_bests = [i for i in filter(lambda x: True, get_second_bests(m))] # not isNaN(x)
        if second_bests == []:
            # print("breaking because second_bests is empty")
            break
        elif all_NaN([second_bests]):
            singleton_rows = [i for i in filter(lambda row: isNaN(second_bests[row]) and not all_NaN([m[row]]), range(N))]
            # print("row indices with one non-NaN value:",singleton_rows)
            # print("breaking because second_bests is all NaN")
            best_pair = (singleton_rows[0],m[singleton_rows[0]].index(0)) # just take the first one for now
            best_pair_found = True

        if not best_pair_found:
            lowest_second_best = min([i for i in filter(lambda x: not isNaN(x), second_bests)])
            # print("lowest_second_best:",lowest_second_best)
            #if isNaN(lowest_second_best): # this was causing problems because we needed to remove NaN before calling min()
                #break

            important_row_indices = [i for i in filter(lambda r: second_bests[r] == lowest_second_best, range(N))] # prioritize the rows with the worst second-bests
            # print("next priority row indices (lowest_second_best = {0}): {1}".format(lowest_second_best,important_row_indices))
            if len(important_row_indices) == 0:
                raise IndexError("No rows selected for priority with matrix {0}.".format(m))
            best_score = -Inf
            # best_pair = None # already initialized

            for important_row_index in important_row_indices:
                important_row = m[important_row_index]
                # print("this important row:",important_row)
                for zero_index in filter(lambda x: important_row[x] == 0, range(M)):
                    # print("this zero index for row {0}: {1}".format(important_row,zero_index))
                    this_eliminated_matrix = eliminate(m,important_row_index,zero_index)
                    # print("eliminating row and col to produce matrix",this_eliminated_matrix)
                    if all_NaN(this_eliminated_matrix):
                        best_pair = (important_row_index,zero_index)
                        # print("breaking because elimination produced all NaNs, adding this pair to solution")
                        break
                    # print("making recursive call\n")
                    subsolution = jap(this_eliminated_matrix) # recursive call; watch out for treachery
                    # print("next subsolution:",subsolution)
                    # print("scoring matrix",this_eliminated_matrix)
                    score = jap_score(this_eliminated_matrix, subsolution)
                    if score > best_score:
                        # print("subsolution accepted")
                        best_score = score
                        best_pair = (important_row_index,zero_index)
                    else:
                        pass # print("subsolution rejected")
            # print("best score found for subproblems:",best_score)

        pairs.append(best_pair)

        m = eliminate(m,best_pair[0],best_pair[1])

    result = [[1 if (i,j) in pairs else 0 for j in range(M)] for i in range(N)] # turn the ordered pairs into sparse matrix

    # take out rows and columns by replacing them with NaN, then repeating procedure until whole matrix is NaN
    # DON'T ever actually remove rows or columns! This will make indices much harder to work with.

    # transpose back if we transposed before solving
    if need_to_transpose:
        result = transpose(result)
    
    # if not is_fully_assigned(result): # note that subproblem results will not be all assigned, so don't actually check for this
        # pass # raise ValueError("Result {0} has left some possible pairs unassigned.".format(result))

    # print("returning result",result,"\n")
    return result

def jap_score(m, sol):
    if all_NaN(m):
        raise ValueError("Matrix is all NaNs.")

    total = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            mm = m[i][j]
            ss = sol[i][j]
            # watch out for NaNs
            if isNaN(mm):
                # skip this element, so we don't have NaN scores for subproblems
                continue
            total += mm*ss

    if total != total: # NaN
        raise ValueError("Score is NaN for matrix {0} and solution {1}.".format(m,sol))
    return total

def jap_solve_and_score(m):
    original_m = [[m[i][j] for j in range(len(m[0]))] for i in range(len(m))]
    sol = jap(m)
    score = jap_score(original_m, sol) # because jap() alters m
    return {"solution":sol,"score":score}

def eliminate(matrix, row, col):
    """
    Replaces rowth row and colth column of matrix with NaNs.
    """
    n_row = len(matrix)
    n_col = len(matrix[0])
    return [[NaN if i == row or j == col else matrix[i][j] for j in range(n_col)] for i in range(n_row)]

def get_second_bests(normalized_matrix):
    """
    Takes a normalized_matrix in which every row has a maximum value of zero.
    Returns a list of the maximum non-zero value in each row.
    """
    if max([max(row) for row in normalized_matrix]) > 0:
        raise ValueError("Matrix must have maximum value of zero.")

    result = []
    for row in normalized_matrix:
        non_zeros = [i for i in filter(lambda x: x != 0, row)]
        if non_zeros == []: # all values in the row are zero; i.e. the same
            result.append(0)
        else:
            non_NaNs = [i for i in filter(lambda x: not isNaN(x), non_zeros)]
            if non_NaNs == []:
                result.append(NaN)
            else:
                result.append(max(non_NaNs))

    return result

# print(get_second_bests([[-1,0,-2],[0,-5,-9],[0,0,0]])) # [-1,-5,0] # works

# print(jap_solve_and_score([[2,3,4,5],[0,0,2,3],[4,1,6,0]])) # [[0,1,0,0],[0,0,0,1],[0,0,1,0]] # works
# print(jap_solve_and_score([[102,103,104,105],[4,4,6,7],[-1,-4,1,-5]])) # [[0,1,0,0],[0,0,0,1],[0,0,1,0]], same as above # works
# print(jap_solve_and_score([[-3,-2,-1,0],[-3,-3,-1,0],[-2,-5,0,-6]])) # [[0,1,0,0],[0,0,0,1],[0,0,1,0]], same as above # works
# print(jap_solve_and_score([[0,-4,-9],[0,-5,-4],[-4,-1,0]])) # [[0,1,0],[1,0,0],[0,0,1]] # works!
# print(jap_solve_and_score([[0,-1,-2,-8],[-1,0,-8,-2],[-6,-8,0,-2],[-8,-2,0,-3]])) # [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]] # works!
# print(jap_solve_and_score([[random.random() for j in range(10)] for i in range(10)])) # something # works! (although haven't checked correctness)

# collect runtimes
runtimes = {}
x = []
import time
import matplotlib.pyplot as plt
try:
    for n in range(1,21):
        t0 = time.time()
        waste = jap_solve_and_score([[random.random() for j in range(n)] for i in range(n)])
        t = time.time()-t0
        if t > 0:
            runtimes[n] = t
            x.append(n)
except KeyboardInterrupt:
    print("stopped in the middle of evaluating for input size",n)
print("runtimes:",runtimes)
# looks like we're exponential here; try getting a polynomial-time solution (the Hungarian algorithm is O(n^4))

import numpy as np
log_runtimes = [math.log(runtimes[n]) for n in runtimes]
exponential_fit = np.polyfit(x, log_runtimes, deg=1)
print(exponential_fit)
beta_1 = exponential_fit[0]
a = math.exp(beta_1)
print("This algorithm runs approximately in {0:.2f}^n time.".format(a))
plt.plot(x,[runtimes[n] for n in x],"r")
plt.plot(x,[math.exp(exponential_fit[1] + exponential_fit[0]*n) for n in x],"b")
plt.show()






