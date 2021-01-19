def set_blocks_cuboid(mc, xyz0, xyz1, block_type):
    # since star unpacking doesn't work e.g. f(*xyz0, *xyz1, b) for a 7-arg function, from within Minecraft forge
    x0,y0,z0 = xyz0
    x1,y1,z1 = xyz1
    mc.setBlocks(x0,y0,z0,x1,y1,z1,block_type)
    # mc.postToChat("{} > {} > {}".format(xyz0, xyz1, block_type))


def get_spiral_xzs(apothem):
    # spiral from center
    # the run lengths are 1,1,2,2,3,3,4,4,... in the four directions
    directions = [(1,0),(0,1),(-1,0),(0,-1)]
    start_xz = (0,0)
    xz = start_xz
    yield start_xz

    step_i = 0
    while True:
        dx = abs(xz[0] - start_xz[0])
        if dx >= apothem:
            break

        direction_i = step_i % 4
        direction = directions[direction_i]
        run_length = step_i // 2 + 1  # 0>1, 1>1, 2>2, 3>2, 4>3, ...
        for run_i in range(run_length):
            x,z = xz
            dir_x, dir_z = direction
            x += dir_x
            z += dir_z
            xz = (x,z)
            yield xz
        step_i += 1


