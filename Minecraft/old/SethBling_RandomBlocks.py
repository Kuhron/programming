import random
 
displayName = "Random Blocks"
 
inputs = (
    ("One Block Per", 16),
    ("Block Id", 0),
    ("Block Data", 0),
    ("Replace Block Id", -1),
)
 
def perform(level, box, options):
    invFreq = options["One Block Per"]
    freq = 1.0 / invFreq
    block = options["Block Id"]
    data = options["Block Data"]
    replace = options["Replace Block Id"]
     
    for x in xrange(box.minx, box.maxx):
        for y in xrange(box.miny, box.maxy):
            for z in xrange(box.minz, box.maxz):
                if replace != -1 and level.blockAt(x, y, z) != replace:
                    continue
                if random.random() < freq:
                    level.setBlockAt(x, y, z, block)
                    level.setBlockDataAt(x, y, z, data)
     
    level.markDirtyBox(box)