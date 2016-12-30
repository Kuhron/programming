# generate "towns" that are nodes on a connected graph
# the only intersections of roads are towns, so if you go one direction from a town, you continue until the next town
# KEY: the towns DO NOT have actual locations in this representation
# you only know what towns are in what directions from each other and how far it is
# no implications about geometry of the space they exist in
# try to learn to navigate the area
# ensure "all roads lead to Rome", i.e. the whole thing should be the same graph

import random
import time

# ensure all roads lead to rome by only adding new towns to existing ones
# store map as class with towns
# you can totally get from A to B and then not be able to go directly back to A

class Map:
    def __init__(self):
        self.n_towns = 0
        self.paths = [] # paths are N, E, S, W. {-1:-1} if no road, {index of next town:travel time} if road
        self.names = []

    def make_town(self):
        name = self.get_name()
        self.names.append(name)
        self.paths.append([])
        self.paths[self.n_towns] = [[-1,-1] for i in range(4)]
        while self.n_towns != 0 and self.paths[self.n_towns] == [[-1,-1] for i in range(4)]:
            # don't do this for the first town until the end
            for i in range(4):
                if random.random() < 0.7:
                    self.paths[self.n_towns][i] = [random.choice(range(self.n_towns)),random.choice(range(1,11))]
        if self.n_towns != 0:
            town, path = self.get_empty_path()
            self.paths[town][path] = [self.n_towns, random.choice(range(1,11))]
        self.n_towns += 1

    def populate(self, n_towns):
        for i in range(n_towns):
            self.make_town()
        # now make sure you can exit the first town
        while self.paths[0] == [[-1,-1] for i in range(4)]:
            for i in range(4):
                if random.random() < 0.7:
                    self.paths[0][i] = [random.choice(range(self.n_towns)),random.choice(range(1,11))]
        town, path = self.get_empty_path()
        self.paths[town][path] = [0, random.choice(range(1,11))]        
        
        # construct list of unreachable towns, to be added into paths afterwards
        self.unreachable_towns = [i for i in range(self.n_towns)]
        for town in self.paths: # list of paths
            for path in town:
                if path[0] in self.unreachable_towns:
                    # a path goes there, so it is reachable
                    self.unreachable_towns.remove(path[0])

        # add them back randomly, although this is quite inefficient
        while self.unreachable_towns != []:
            town, path = self.get_empty_path()
            destination = random.choice([i for i in filter(lambda x: x != self.paths.index(town), self.unreachable_towns)])
            self.paths[town][path] = [destination, random.choice(range(1,11))]
            self.unreachable_towns.remove(destination)

        # just leave them there for now, but don't start off in one
        self.reachable_towns = [i for i in filter(lambda x: x not in self.unreachable_towns, [i for i in range(self.n_towns)])]

    def get_empty_path(self):
        while True:
            town = random.choice(range(self.n_towns))
            path = random.choice(range(4))
            if -1 in self.paths[town][path]:
                return town, path


    name_components = {
        "preword":["New","San","Saint","Mount","Old"],
        "prefix":["South","East","North","West","Bridge","Lake","River","Valley","Isle","Field","Ridge","Home","Sun","Night","Day","German",
                  "French","Dutch","Oak","Birch","Sea","Gray","White","Black","Red","Green"],
        "name":["Jefferson","Manhattan","Lincoln","Hyde","Washington","Wilson","Wall","Poplar","Benton","Joseph","Abraham","Hamilton","Jackson",
                "Olympia","McKinley","Henry","Sawyer","Thompson","Henderson","Pierce"],
        "suffix":["port","ville","town","brook","dale","vale","side","ton","shire","land","wood","crest","view","field","lock"],
        "postword":["City","Park","Creek","Trail","Shore","Beach","Harbor","Bay"]
    }

    def get_name(self):
        name_components = self.name_components
        name = ""
        add_postword = False

        if random.random() < 0.2:
            name += random.choice(name_components["preword"]) + " "

        if random.random() < 0.5:
            name += random.choice(name_components["prefix"])
            add_postword = True # will need postword if the mid-name has only a prefix with no suffix
        else:
            name += random.choice(name_components["name"])

        if random.random() < 0.4:
            name += random.choice(name_components["suffix"])
            add_postword = False

        if add_postword or random.random() < 0.3:
            name += " " + random.choice(name_components["postword"])

        if name in self.names:
            return self.get_name() # try again. doing this recursively is probably a bad idea, but whatever
        else:
            return name

def show_time(amount):
    t0 = time.time()
    while time.time()-t0 < amount:
        print("Elapsed: {0:.2f}".format(time.time()-t0),end="\r")
        time.sleep(0.01)

M = Map()
# for i in range(15):
#     M.make_town()
M.populate(int(input("How many towns would you like to navigate? ")))

# the game: visit every town and then find your way back to the original town (ensure that it is reachable)

start_point = random.choice(M.reachable_towns)
current_point = start_point
remaining_towns = M.reachable_towns[:]
remaining_towns.remove(current_point)
remaining_towns_finished = False
steps = 0
start_time = time.time()

while True:
    if current_point in remaining_towns:
        remaining_towns.remove(current_point)
    if (not remaining_towns_finished) and remaining_towns == []:
        print("You've visited all the towns! Now make it back to {0}.".format(M.names[start_point]))
        remaining_towns_finished = True
    if remaining_towns_finished and current_point == start_point:
        print("\nYou are now in {0}.".format(M.names[current_point]))
        print("Congratulations! You navigated everything in {0} steps and {1:.2f} seconds.".format(steps,time.time()-start_time))
        break

    direction_input = input("\nYou are now in {0}. Go N, E, S, or W? ".format(M.names[current_point]))

    if direction_input in ["N","E","S","W"]:
        direction = ["N","E","S","W"].index(direction_input)
    elif direction_input in ["8","6","2","4"]: # can also use numpad
        direction = ["8","6","2","4"].index(direction_input)
    else:
        print("That is not a valid direction.")
        continue

    if -1 in M.paths[current_point][direction]:
        print("There is no road that goes {0} from {1}.".format(["N","E","S","W"][direction],M.names[current_point]))
        continue
    else:
        print("Going {0} from {1}.".format(["N","E","S","W"][direction],M.names[current_point]))
        show_time(M.paths[current_point][direction][1])
        print("Done. Elapsed: {0:d}".format(M.paths[current_point][direction][1]))
        current_point = M.paths[current_point][direction][0]

    steps += 1















