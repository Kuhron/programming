import random

class Board:
    # store data as sets of integer possibilities when unknown, as strings when known

    def __init__(self):
        self.data = [[set(range(1,10)) for i in range(9)] for i in range(9)]
        self.fill_data()
        self.show()
        self.solve()

    def fill_data(self): 
        print("For each row, enter the numbers as a 9-digit input, with 0 denoting unknown cells.")
        for r in range(9):
            imp = input()
            if len(imp) != 9:
                print("invalid input")
                break
            imp = [i for i in imp]
            for i in range(9):
                if imp[i] != "0":
                    self.data[r][i] = imp[i]

    def show(self):
        d = self.data
        for r in range(9):
            for c in range(9):
                if type(d[r][c]) != set:
                    if c in [2,5]:
                        print("%s |" % d[r][c], end = " ")
                    else:
                        print("%s" % d[r][c], end = " ")
                else:
                    if c in [2,5]:
                        print("  |", end = " ")
                    else:
                        print(" ", end = " ")
            print()
            if r in [2,5]:
                print("-"*21)

    def get_row(self, n):
        return self.data[n]

    def get_column(self, n):
        return [self.data[r][n] for r in range(9)]

    def get_box(self, r, c):
        first_r = int(r/3)
        first_c = int(c/3)
        return [self.data[rr][cc] for cc in range(first_c, first_c+3) for rr in range(first_r, first_r+3)]

    def is_solved(self):
        d = self.data
        for r in range(9):
            for c in range(9):
                if type(d[r][c]) == set:
                    return False
        return True

    def solve(self):
        # progress_made = False
        # for r in range(9):
        #   for c in range(9):
        #         # do stuff
        # if progress_made:
        #   self.solve()
        limiter = 0
        while not self.is_solved() and limiter < 10**6:
            cell = self.pick_cell()
            self.solve_cell(cell)
            #self.show()
            limiter += 1
        if limiter >= 10**6:
            print("This is as far as I got.")
            self.show()
        if self.is_solved():
            print("Done!")
            self.show()

    def pick_cell(self):
        return [random.choice(range(9)),random.choice(range(9))]

    def solve_cell(self,cell):
        # needs more logic and inference, currently just waits until there is only one possibility for a cell
        
        r_cell = cell[0]
        c_cell = cell[1]

        # prevent it from printing when it gets a cell that is already known
        if len(self.data[r_cell][c_cell]) == 1:
            return

        row = self.get_row(r_cell)
        column = self.get_column(c_cell)
        box = self.get_box(r_cell, c_cell)
        all_taken = row+column+box
        for i in range(9):
            if i in all_taken:
                self.data[r_cell][c_cell].remove(i)
        if len(self.data[r_cell][c_cell]) == 1:
            self.data[r_cell][c_cell] = str(list(self.data[r_cell][c_cell])[0])
            self.show()

B = Board()