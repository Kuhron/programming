# ecosystem simulation
# include various types of creatures
# there should be abiotic factors in addition to the creatures
# creatures should have sizes, move, eat, excrete, reproduce, and die
# play around with it and see if you can arrive at any self-sustaining models
# for best results, use a cage (walled finite area) or toroidal array, rather than things being obliterated by walking off the edge

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

def is_empty(array,x_1,x_2,y_1,y_2):
    array = np.array(array)
    area = array[x_1:x_2,y_1:y_2]
    if np.sum(area) != 0:
        return False
    if area.size == 0:
        return False
    return np.max(area) == 0

def environment_density(x,y):
    # return 0
    # return math.floor((x/3.0+y/3.0)%2)*2
    return 6 if random.random()<0.35 and ((x+y) % 8 == 3 or (x-y) % 8 == 5) else 0

class Environment: # the space in which things exist and interact
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.area = w*h
        self.densities = np.array([[environment_density(xx,yy) for xx in range(w)] for yy in range(h)])

    def find_space(self,w,h):
        # finds empty space of dimensions x by y at random
        for i in range(int((self.w*self.h)/(w*h))): # only try so many times, the number of times the desired area can fit into the environment
            x_, y_ = [random.randrange(self.w-w+1) for i in range(2)] # pick the first corner of the area to check
            if is_empty(self.densities,x_,x_+w,y_,y_+h):
                return x_,y_
        return -1

    # the world is a 2D grid, with each square colored based on density
    # 0 = white (nothing is there), 1 = yellow, 2 = green, 3 = blue, 4 = purple, 5 = red, 6 = black (max)

    def show(self):
        #densities = [[(xx+yy) % 7 for xx in range(self.x)] for yy in range(self.y)] # later, change this to show the creatures and chemicals
        densities = self.densities

        # http://stackoverflow.com/questions/7229971/2d-grid-data-visualization-in-python
        cmap = matplotlib.colors.ListedColormap(["white","yellow","green","blue","purple","brown","black"])
        bounds = [0,1,2,3,4,5,6,7]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(densities, interpolation="nearest", cmap=cmap, norm=norm)
        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds)
        plt.show()

e = Environment(100,100)

class Occupant: # anything biotic or abiotic in the environment
    def __init__(self):
        environment = e # THERE CAN BE ONLY ONE

class Creature(Occupant): # biotic
    living = True # class-level variable
    creatures = {"mouse":[], "dog":[], "bear":[]}

    def __init__(self, name):
        Occupant.__init__(self)
        self.name = name
        self.spawned = False
        self.x = None
        self.y = None
        self.location = [self.x,self.y]

    def spawn(self):
        if self.spawned:
            return False
        if not (random.random() < self.frequency): # only actually spawn them with their species's frequency
            return False
        space = e.find_space(self.w, self.h)
        if space == -1:
            # no space was found
            return False
        x,y = space
        self.x = x
        self.y = y
        e.densities[x:x+self.w,y:y+self.h] = self.color
        Creature.creatures[self.species].append(self) # add this instance to the directory
        self.spawned = True
        return True

    def move(self):
        old_x = self.x
        old_y = self.y
        directions = ["x+","x-","y+","y-"]
        while directions != []: # only try every possibility, no more
            d = random.choice(directions)
            if d[0] == "x": # move along x-axis
                new_y = self.y
                new_y_1 = self.y
                new_y_2 = self.y+self.h
                if d[1] == "+": # move right
                    new_x = self.x + 1
                    new_x_1 = self.x + self.w
                    new_x_2 = self.x + self.w + 1
                else: # move left
                    new_x = self.x - 1
                    new_x_1 = self.x - 1
                    new_x_2 = self.x # - 1 + 1
            else: # move along y-axis
                new_x = self.x
                new_x_1 = self.x
                new_x_2 = self.x+self.w
                if d[1] == "+": # move up (I think?)
                    new_y = self.y + 1
                    new_y_1 = self.y + self.h
                    new_y_2 = self.y + self.h + 1
                else:
                    new_y = self.y - 1
                    new_y_1 = self.y - 1
                    new_y_2 = self.y # - 1 + 1
            # print("{0} got to this point".format(self.name))
            if is_empty(e.densities,new_x_1,new_x_2,new_y_1,new_y_2):
            #if is_empty(e.densities,new_x+self.w-1,new_x+self.w,new_y+self.h-1,new_y+self.h):
                # make sure only to check the cells that will be moved into
                # do not check the cells that are currently occupied, even though they will stay occupied! it makes is_empty always false
                self.x = new_x
                self.y = new_y
                self.update_position_in_environment(old_x,new_x,old_y,new_y)
                # print("{0} has moved!".format(self.name))
                return # better than break because now the function exits
            else:
                directions.remove(d)
        if True: # if you're reading this, program
            print("{0} failed to move".format(self.name))

    def update_position_in_environment(self,old_x,new_x,old_y,new_y):
        e.densities[old_x:old_x+self.w,old_y:old_y+self.h] = 0
        e.densities[new_x:new_x+self.w,new_y:new_y+self.h] = self.color

class Bear(Creature):
    def __init__(self, name):
        Creature.__init__(self, name)
        self.species = "bear"
        self.w = 3
        self.h = 3
        self.color = 5
        self.frequency = 0.05

class Dog(Creature):
    def __init__(self, name):
        Creature.__init__(self, name)
        self.species = "dog"
        self.w = 2
        self.h = 2
        self.color = 4
        self.frequency = 0.2

#d1 = Dog("Fido")
#print(d1.name)
#print(d1.species)

class Mouse(Creature):
    def __init__(self, name):
        Creature.__init__(self, name)
        self.species = "mouse"
        self.w = 1
        self.h = 1
        self.color = 2
        self.frequency = 0.6

class Chemical(Occupant): # abiotic
    living = False # class-level variable
    chemicals = {"iron":[]}

    def __init__(self, name):
        Occupant.__init__(self)
        self.name = name

class Iron(Chemical):
    def __init__(self, name):
        Chemical.__init__(self, name)
        self.compound = "iron"

#i1 = Iron("1")
#print(i1.name)
#print(i1.compound)

for i in range(40):
    if not Dog(name="dog_"+str(i)).spawn(): # new dog each time
        pass #print("Dog %s failed to spawn." % i)
    if not Mouse(name="mouse_"+str(i)).spawn():
        pass #print("Mouse %s failed to spawn." % i)
    if not Bear(name="bear_"+str(i)).spawn():
        pass #print("Bear %s failed to spawn." % i)
e.show()

#print(Creature.creatures)

def move_creatures():
    for spec in sorted(Creature.creatures,reverse=True):
        # print("now moving",spec)
        for indiv in Creature.creatures[spec]:
            indiv.move()

for i in range(1000):
    move_creatures()
e.show()