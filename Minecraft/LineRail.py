import random


SAVE_DIR = "C:/Users/Wesley/AppData/Roaming/.minecraft/saves/"
LOADNAME = SAVE_DIR + "PythonExperimentation"

# Where do you want to start the rail?
# START_X, START_Y, and START_Z are the map coordinates
# This is the position of the first rail block
START_X = random.randint(-1000, 1000)
START_Y = 69
START_Z = random.randint(-1000, 1000)

DIRECTION = random.choice("+-") + random.choice("XZ")
DISTANCE = random.randint(50, 200)
TUNNELHEIGHT = 3
POWERSPACING = 26
BEDMAT = 1
BEDDATA = 0
PILLARSPACING = 8
PILLARMAT = 1
PILLARDATA = 0
LIGHTSPACING = 8
LIGHTMAT = 50
LIGHTDATA = 5  # Default, 5 (torch placed on the ground)

# Do a rough recalculation of the lighting and heightmap?
# Slows it down to do a rough re-light.
# True  do the rough fix (can take a while)
# False don't bother (tunnels will be black inside and heightmap not updated)
LIGHTINGFIX = True
# NOTE: if you set LIGHTINGFIX = False,
# consider re-lighting the map in a tool like MCEdit

# Do you want a bunch of info on what the script is doing?
# True enables verbose data output
# False minimal text info, slight speed increase
VERBOSE = True

##############################################################
#  Don't edit below here unless you know what you are doing  #
##############################################################

# input filtering
DIRECTION = DIRECTION.upper()
if DIRECTION not in ('+X','-X','+Z','-Z'):
    print("DIRECTION value '" + str(DIRECTION) + "' is not a valid choice, please use '+X', '-X', '+Z', or '-Z'")
if DISTANCE < 1:
    DISTANCE = 1
    print('DISTANCE was less than 1. Setting to 1 and continuing.')
if POWERSPACING < 1:
    POWERSPACING = 1
    print('POWERSPACING was less than 1. Setting to 1 and continuing.')
if TUNNELHEIGHT < 1:
    TUNNELHEIGHT = 1
    print('TUNNELHEIGHT was less than 1. Setting to 1 and continuing.')
if PILLARSPACING < 1:
    PILLARSPACING = 0
    print('PILLARSPACING was less than 1. Building no pillars')
if LIGHTSPACING < 1:
    LIGHTSPACING = 0
    print('LIGHTSPACING was less than 1. Adding no lights')

if BEDMAT in (0,6,8,9,10,11,12,13,18,20,26,27,28,30,37,38,39,40,44,
              50,51,52,53,55,59,63,64,65,66,67,68,69,70,71,72,75,76,
              77,78,80,81,83,85,90,92,93,94):
    print("a BEDMAT value of '" + str(BEDMAT) + "' is not generally a good idea. Just saying.")

# assemble the material dictionaries
BEDINFO = {'B':BEDMAT,'D':BEDDATA}
PILLARINFO = {'B':PILLARMAT,'D':PILLARDATA}
LIGHTINFO = {'B':LIGHTMAT,'D':LIGHTDATA}
POWERINFO = {'B':76,'D':5}
AIRINFO = {'B':0,'D':0}

LIGHT_REDUCTION_DICT = {0:0, 8:2, 18:1, 20:0, 27:0, 28:0, 37:0, 38:0, 39:0, 40:0,
                        50:0, 51:0, 52:0, 63:0, 64:0, 65:0, 66:0, 68:0, 69:0,
                        70:0, 71:0, 72:0, 75:0, 76:0, 77:0, 78:1, 79:2, 83:1, 85:0}
LUMINANCE_DICT = {51:15, 91:15, 10:15, 11:15, 89:15, 95:15, 50:14, 62:13,
                  90:11, 74:9, 94:9, 76:7}
# absolute vertical limits of the map
MAPTOP = 127
MAPBTM = 0

# The following is an interface class for .mclevel data for minecraft savefiles.
# The following also includes a useful coordinate to index convertor and several
# other useful functions.

import Minecraft.mcInterface as mcInterface

# This is the end of the MCLevel interface.

# Now, on to the actual code.

def get_surface(X,Y,Z, mclevel):
    '''Return the Y position of the highest non-air block at or below X,Y,Z.'''
    get_block = mclevel.block
    while Y > MAPBTM:
        info = get_block(X,Y,Z)
        if info is None: return None
        if info['B'] == 0:
            Y -= 1
            continue
        break
    return Y

def calc_column_lighting(X, Y, Z, mclevel):
    '''Recalculate the sky lighting of the column.'''
    
    # Begin at the top with sky light level 15.
    cur_light = 15
    # traverse the column until cur_light == 0
    # and the existing light values are also zero.
    get_block = mclevel.block
    set_block = mclevel.set_block
    get_height = mclevel.retrieve_heightmap
    set_height = mclevel.set_heightmap
    #get the current heightmap
    cur_height = get_height(X,Z)
    # if this doesn't exist, the block doesn't exist either, abort.
    if cur_height is None: return None
    # Start at the given Y, or at the current height, whichever is greater
    Y = max((cur_height, Y))
    # set a flag that the highest point has been updated
    height_updated = False
    
    while True:
        #get the block sky light and type
        block_info = get_block(X,Y,Z,'BS')
        block_light = block_info['S']
        block_type = block_info['B']
        #set the new light_reduction
        if block_type in LIGHT_REDUCTION_DICT:
            # partial light reduction
            light_reduction = LIGHT_REDUCTION_DICT[block_type]
        else:
            # full light reduction
            light_reduction = 16
        # update the height map if it hasn't been updated yet,
        # and the current block reduces light
        if (not height_updated) and (light_reduction != 0):
            new_height = Y + 1
            if new_height == 128: new_height = 127
            set_height(X,new_height,Z)
            height_updated = True
        #compare block with cur_light, escape if both 0
        if block_light == 0 and cur_light == 0: break
        #set the block light if necessary
        if block_light != cur_light:
            set_block(X,Y,Z,{'S':cur_light})
        # decrement the current light level
        cur_light += -light_reduction
        if cur_light < 0: cur_light = 0
        #increment and check Y
        Y += -1
        if Y < 0:
            # we reached the bottom without running out of light
            if (not height_updated):
                # we reached the bottom without hitting anything!
                new_height = Y + 1
                set_height(X,new_height,Z)
            break

def add_light(light_to_add, X, Y, Z, mclevel):
    '''add a local lighting source to mclevel at (X, Y, Z) with value light_to_add'''
    # localize the block get and set calls
    get_block = mclevel.block
    set_block = mclevel.set_block
    # populate the different light level sets
    all_lights = []
    for i in range(light_to_add+1):
        new_set = set()
        all_lights += [new_set]
    # cur_idx is the current light level minus one
    cur_idx = light_to_add - 1
    # the top light level set should contain only the origin block
    top_level = all_lights[cur_idx]
    top_level.add((X,Y,Z))
    # big old loop!
    while True:
        cur_light = cur_idx + 1
        for coord in all_lights[cur_idx]:
            X, Y, Z = coord
            # check the current light level
            this_light_was = get_block(X,Y,Z,'L')['L']
            # if it's at or above the current light, skip it
            if this_light_was >= cur_light: continue
            # if it's below, set it, and keep going
            set_block(X,Y,Z,{'L':cur_light})
            # now see if any other blocks should be lit
            adjacent_blocks = [(X+1,Y,Z),
                               (X-1,Y,Z),
                               (X,Y+1,Z),
                               (X,Y-1,Z),
                               (X,Y,Z+1),
                               (X,Y,Z-1),]
            for other_coord in adjacent_blocks:
                # we don't need X, Y, or Z anymore, so re-use them here
                X, Y, Z = other_coord
                # check the light level above this one
                # to avoid the most common unneccessary growth
                if other_coord in all_lights[cur_idx+1]: continue
                # this is the type of the other block, adjacent to coord
                this_type = get_block(X,Y,Z)['B']
                if this_type not in LIGHT_REDUCTION_DICT:
                    # the block is opaque, skip it
                    continue
                else:
                    reduction = 1 + LIGHT_REDUCTION_DICT[this_type]
                new_light_level = cur_light - reduction
                # the block lighting would be too low, skip it
                if new_light_level < 1: continue
                # add the block to the appropriate set
                appropriate_idx = new_light_level - 1
                appropriate_set = all_lights[appropriate_idx]
                appropriate_set.add(other_coord)
            
        cur_idx += -1
        if cur_idx == 0: break

def lay_the_rail(mclevel):
    '''Increment over the rail positions and call the appropriate create functions when needed.'''
    # some more useful globals
    RAIL_DATA_DICT = {'X':1, 'Z':0}
    RAIL_DATA = RAIL_DATA_DICT[DIRECTION[1]]
    INCREMENT_DICT = {'-':-1, '+':1}
    INCREMENT = INCREMENT_DICT[DIRECTION[0]]
    # the position vector is [X, Y, Z]
    DIRECTION_AXIS_DICT = {'X':0, 'Z':2}
    DIRECTION_AXIS = DIRECTION_AXIS_DICT[DIRECTION[1]]
    SIDE_DIRECTION_AXIS_DICT = {'X':2, 'Z':0}
    SIDE_DIRECTION_AXIS = SIDE_DIRECTION_AXIS_DICT[DIRECTION[1]]
    # localize set block
    set_block = mclevel.set_block
    # make a lighting update list
    light_emit_list = []
    # X is 0 and Z is 2 in the position list
    position = [START_X,START_Y,START_Z]
    for total_dist in range(DISTANCE):
        X, Y, Z = position
        # find the existing height of the land
        top_y = Y+TUNNELHEIGHT-1
        surface_height = get_surface(X,top_y,Z,mclevel)
        if surface_height is None:
            print("The rail ran off the edge of the map or something!")
            break
        # place the rail bed
        set_block(X,Y-1,Z,BEDINFO)
        # if the rail bed emits light, add it to the light_emit_list
        if BEDMAT in LUMINANCE_DICT: light_emit_list += [(X,Y,Z)]
        # place the lighting
        if LIGHTSPACING and (total_dist % LIGHTSPACING == 0):
            # copy the current position
            side_pos = position[:]
            # put it on the opposite side as the rail power
            side_pos[SIDE_DIRECTION_AXIS] += -INCREMENT
            # set the light block
            set_block(side_pos[0],side_pos[1],side_pos[2],LIGHTINFO)
            # add the light block to the light emit list
            light_emit_list += [(side_pos[0],side_pos[1],side_pos[2])]
            # move the position to the block below the light
            side_pos[1] += -1
            # and add a light support block
            set_block(side_pos[0],side_pos[1],side_pos[2],BEDINFO)
        # place the rail
        this_rail_data = RAIL_DATA
        if (POWERSPACING and total_dist % POWERSPACING == 0):
            # set the rail value
            rail_value = 27
            #turn the rail on
            this_rail_data += 8
            side_pos = position[:]
            # place the redstone torch
            side_pos[SIDE_DIRECTION_AXIS] += INCREMENT
            set_block(side_pos[0],side_pos[1],side_pos[2],POWERINFO)
            # add the torch to the lighting update list
            light_emit_list += [(side_pos[0],side_pos[1],side_pos[2])]
            # place the block supporting the redstone torch
            side_pos[1] += -1
            set_block(side_pos[0],side_pos[1],side_pos[2],BEDINFO)
            # re-light, if required:
            if LIGHTINGFIX:
                calc_column_lighting(side_pos[0],side_pos[1]+1,side_pos[2], mclevel)
        else:
            # place a normal rail block
            rail_value = 66
        rail_info = {'B':rail_value, 'D':this_rail_data}
        # actually place the rail block
        set_block(X,Y,Z,rail_info)
        # place the tunnel or pillars
        if surface_height > Y:
            # make a tunnel
            if VERBOSE:
                print("Position " + str(position) + " is a tunnel")
            for cur_y in range(Y+1, Y+TUNNELHEIGHT):
                set_block(X,cur_y,Z,AIRINFO)
        # if you don't make a tunnel, check if you need supports
        elif (surface_height < Y-2 and
              PILLARSPACING and
              (total_dist % PILLARSPACING == 0)):
            #make pillars
            if VERBOSE:
                print("Position " + str(position) + " has a pillar")
            for cur_y in range(surface_height,Y-1):
                set_block(X,cur_y,Z,PILLARINFO)
        else:
            if VERBOSE:
                print("Position " + str(position) + " is just normal")
        # re-light, if required:
        if LIGHTINGFIX:
            calc_column_lighting(X, Y, Z, mclevel)
        # increment position
        position[DIRECTION_AXIS] += INCREMENT
    # when we're all done, return the list of blocks that emit light
    return light_emit_list

def calc_all_emmission_lights(light_list, mclevel):
    '''do the emission lighting updates'''
    get_block = mclevel.block
    for pos in light_list:
        if VERBOSE:
            print("Lighting position: " + str(pos))
        X,Y,Z = pos
        block = get_block(X,Y,Z)['B']
        # if this block does not emit light, go to the next one
        if block not in LUMINANCE_DICT: continue
        starting_light = LUMINANCE_DICT[block]
        add_light(starting_light, X, Y, Z, mclevel)

def main(the_map):
    '''Load the file, create the rail line, and save the new file.
    '''
    print("Laying the rail")
    lights = lay_the_rail(the_map)
    return lights

def standalone():
    print("Importing the map")
    try:
        the_map = mcInterface.SaveFile(LOADNAME)
    except IOError:
        print('File name invalid or save file otherwise corrupted. Aborting')
        return None
    lights = main(the_map)
    if LIGHTINGFIX:
        print("propigating lighting (can take a while)")
        calc_all_emmission_lights(lights, the_map)
    print("Saving the map (can also take a while)")
    the_map.write()
    if VERBOSE:
        print("finished")
        input("press Enter to close")

if __name__ == '__main__':
    standalone()
    
# Needed updates:
