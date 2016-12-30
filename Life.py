import random, time, tkinter

X = 60
Y = 60

first_grid = [[0 for i in range(X)] for j in range(Y)]

# ROW-MAJOR
# 1ST INDEX WHEN GETTING ELEMENTS IS THE Y COORDINATE, THE ROW
# 2ND INDEX WHEN GETTING ELEMENTS IS THE X COORDINATE, THE COLUMN
# 1ST EXPRESSION IN LIST COMPREHENSION IS X VARIABLE, I, SINCE YOU ARE GOING ACROSS A ROW
# 2ND EXPRESSION IN LIST COMPREHENSION IS Y VARIABLE, J, SINCE YOU ARE GOING DOWN THE ROWS

def spawn_glider(grid):
    # ignores whether the glider overlaps anything that is already there
    x = random.randint(1,X-2)
    y = random.randint(1,Y-2)
    glider = [
        [0,1,0],
        [0,0,1],
        [1,1,1]
    ]
    if random.random() < 0.5:
        # horizontal flip
        glider = [[r[2-i] for i in range(3)] for r in glider]
    if random.random() < 0.5:
        # vertical flip
        glider = [glider[2-i] for i in range(3)]
    if random.random() < 0.5:
        # diagonal flip along x=y
        # THIS PART IS BACKWARDS FROM ONE AND ONLY ONE OF THE ROW-MAJOR CONVENTIONS BECAUSE IT IS FLIPPING ROWS AND COLUMNS
        glider = [[glider[j][i] for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            grid[y+j-1][x+i-1] = glider[j][i]
    return grid

def neighbors(grid,x,y):
    n = 0
    for i in range(-1,2):
        for j in range(-1,2):
            if (i == 0 and j == 0) or x+i >= X or x+i < 0 or y+j >= Y or y+j < 0:
                continue
            if grid[y+j][x+i] == 1:
                n += 1
    return n

def evolve(grid):
    # only place the next live cells, for simplicity; default to dead
    new_grid = [[0 for i in range(X)] for j in range(Y)]
    for r in range(Y):
        for c in range(X):
            if grid[r][c] == 1 and neighbors(grid,c,r) in [2,3]:
                new_grid[r][c] = 1
            elif grid[r][c] == 0 and neighbors(grid,c,r) == 3:
                new_grid[r][c] = 1
    return new_grid

def show_grid_jank(grid):
    # jank af
    #print("\n"*20)
    for r in grid:
        for c in r:
            print(" X"[c], end = "")
        print()

g = first_grid
root = tkinter.Tk()
# for r in range(Y):
#     for c in range(X):
#         tkinter.Label(root, text=g[r][c],borderwidth=1).grid(row=r,column=c)
#grid_text = tkinter.StringVar()
#tkinter.Label(root, text=grid_text,borderwidth=1).grid(row=Y,column=X)
#root.mainloop()
for i in range(10**5):
    # display this grid
    show_grid_jank(g)
    #grid_text.set("\n".join([" ".join([str(c) for c in r]) for r in g]))
    # update grid
    if random.random() < 0.1:
        g = spawn_glider(g)
    g = evolve(g)
    time.sleep(0.005)