from mcpi.minecraft import Minecraft
import mcpi.block as block
import math


def get_hall_direction(player_direction):
    # ignore y coordinate
    facing_x, _, facing_z = player_direction
    # change z to y so I can think about tangent correctly
    x, y = facing_x, facing_z
    # below y=x line means y < x, above means y > x
    if y < x:
        # we are facing either positive x or negative y
        # now compare position to the y=-x line, if above it (y > -x) then we are facing right in xy plane (positive x)
        if y < -x:
            # facing negative y, below both diagonal lines
            new_x, new_y = 0, -1
        else:
            # facing positive x
            new_x, new_y = 1, 0
    else:
        # we are facing either positive y or negative x
        # if above line y=-x then we are above both diagonals, so facing positive y
        if y > -x:
            # positive y
            new_x, new_y = 0, 1
        else:
            # negative x
            new_x, new_y = -1, 0
    # now return vector where new_y is the z coord
    return (new_x, 0, new_y)


def get_hall_starting_point(player_pos, hall_direction):
    # start it a little ahead of the player
    steps_ahead = 2
    x_raw, y_raw, z_raw = player_pos  # floats
    x, y, z = int(x_raw), int(y_raw), int(z_raw)
    dx, dy, dz = hall_direction
    dx *= steps_ahead
    dy *= steps_ahead  # should always be 0 but whatever
    dz *= steps_ahead
    return (x+dx, y+dy, z+dz)


def get_hall_block_positions(player_pos, direction):
    hall_direction = get_hall_direction(direction)
    hall_starting_point = get_hall_starting_point(player_pos, hall_direction)

    max_length = 5000
    hall_height = 8
    hall_width = 3  # between walls
    assert hall_width % 2 == 1
    d_perpendicular_magnitude = (hall_width + 1) / 2  # how much to move the walls either direction, perpendicular to the wall propagation axis
    x,y,z = hall_starting_point
    dx,dy,dz = hall_direction
    assert dy == 0
    assert abs(dx) + abs(dz) == 1
    if dx != 0:
        assert dz == 0
        propagation_direction = "x"
        perpendicular_direction = "z"
    else:
        assert dz != 0
        propagation_direction = "z"
        perpendicular_direction = "x"

    step_i = 0
    while True:
        if step_i % 100 == 0:
            mc.postToChat("hall length is now {}".format(step_i))
        this_dx = dx * step_i
        this_dy = dy * step_i
        this_dz = dz * step_i
        this_x = x + this_dx
        this_y = y + this_dy
        this_z = z + this_dz
        if perpendicular_direction == "x":
            perpendicular_changes = [(-d_perpendicular_magnitude, 0, 0), (d_perpendicular_magnitude, 0, 0)]
        elif perpendicular_direction == "z":
            perpendicular_changes = [(0, 0, -d_perpendicular_magnitude), (0, 0, d_perpendicular_magnitude)]
        else:
            raise Exception("invalid perpendicular direction")

        y_bottom = y
        y_top = y_bottom + hall_height

        for perpendicular_change in perpendicular_changes:
            cuboid_x0 = this_x + perpendicular_change[0]
            cuboid_y0 = y_bottom
            cuboid_z0 = this_z + perpendicular_change[2]
            cuboid_x1 = cuboid_x0
            cuboid_y1 = y_top
            cuboid_z1 = cuboid_z0
            yield (cuboid_x0, cuboid_y0, cuboid_z0, cuboid_x1, cuboid_y1, cuboid_z1)

        step_i += 1

        if step_i >= max_length:
            break


def make_hall(player_pos, direction, block_id):
    for x0, y0, z0, x1, y1, z1 in get_hall_block_positions(player_pos, direction):
        mc.setBlocks(x0,y0,z0,x1,y1,z1,block_id)


if __name__ == "__main__":
    mc = Minecraft()

    player_pos = mc.player.getPos()
    direction = mc.player.getDirection()  # unit vector in 3d, which direction the player is facing
    
    mc.postToChat("position is {}".format(player_pos))
    mc.postToChat("direction is {}".format(direction))

    make_hall(player_pos, direction, block.WOOL_YELLOW)

