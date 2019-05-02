import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

# generate random grains of sand with nega moment vectors
# see what kinds of configurations will result in strong fields
# want randomly distributed grains to have zero field on average
# want something like maximum cross product of moments to create large field

class Grain:
    def __init__(self, position, moment):
        position = np.array(position)
        self.position = position
        self.moment = moment

class Moment:
    def __init__(self, vector):
        vector = np.array(vector)
        self.vector = vector
        self.magnitude = magnitude(self.vector)
        self.direction = self.vector / self.magnitude if self.magnitude != 0 else np.array([0, 0, 0])

    def __eq__(self, other):
        for a, b in zip(self.vector, other.vector):
            if not float_equal(a, b):
                return False
        return True

    def is_zero(self):
        return self == Moment((0, 0, 0))


def float_equal(a, b):
    return abs(a - b) < 1e-9

def deg(x):
    return x * 180/np.pi

def rad(x):
    return x * np.pi/180

def magnitude(v):
    return np.linalg.norm(v)

def distance(a, b):
    return np.linalg.norm(a - b)

def get_effective_moment_at_reference_point(grain, reference_point):
    moment = grain.moment
    r = distance(grain.position, reference_point)
    factor = r**-2  # try 1/r^2, 1/r^3, maybe some others
    return Moment(moment.vector * factor)

def rand_unif_3d_sphere():
    while True:
        v = np.random.uniform(-1, 1, size=3)
        if magnitude(v) <= 1:
            return v

def get_grains(n):
    # randomly distributed inside a unit sphere
    result = []
    while len(result) < n:
        position = rand_unif_3d_sphere()
        if magnitude(position) <= 1:
            moment = Moment(rand_unif_3d_sphere())
            g = Grain(moment=moment, position=position)
            result.append(g)
    return result

def get_field_at_reference_point(grains, reference_point):
    moments = [get_effective_moment_at_reference_point(g, reference_point) for g in grains]
    return moment_multiply_array(moments)

def moment_multiply(a, b):
    # get weighted average of direction vectors
    # extend that direction to the line connecting the endpoints of a and b
    # vector times zero moment should just be that vector

    if a.is_zero():
        return b
    if b.is_zero():
        return a

    l_a = a.magnitude
    l_b = b.magnitude
    new_direction = (l_a * a.direction + l_b * b.direction) / (l_a + l_b)
    phi = angle(new_direction, b.vector)

    phi_diff = phi - angle(new_direction, b.direction)
    assert float_equal(phi_diff, 0), phi_diff
    theta = angle(a.vector, b.vector)
    theta_diff = theta - angle(a.direction, b.direction)
    assert float_equal(theta_diff, 0), theta_diff
    theta_minus_phi = theta - phi
    theta_minus_phi_diff = theta_minus_phi - angle(new_direction, a.direction)
    assert float_equal(theta_minus_phi_diff, 0), theta_minus_phi_diff

    l_ab = distance(a.vector, b.vector)
    sin_ratio = math.sin(theta) / l_ab
    sin_alpha = sin_ratio * l_a
    sin_beta = sin_ratio * l_b

    alpha = math.asin(sin_alpha)
    # new_length is opposite alpha, adjacent to phi
    # b is between phi and alpha
    third_angle = np.pi - phi - alpha
    new_sin_ratio = math.sin(third_angle) / l_b
    # assert float_equal(new_sin_ratio * (l_ab / 2), phi)
    new_length = sin_alpha / new_sin_ratio

    beta = math.asin(sin_beta)
    # new length is opposite beta, adjacent to (theta - phi)
    # a is between (theta - phi) and beta
    third_angle_from_beta = np.pi - theta_minus_phi - beta
    new_sin_ratio_from_beta = math.sin(third_angle_from_beta) / l_a
    # assert float_equal(new_sin_ratio_from_beta * (l_ab / 2), theta_minus_phi)
    new_length_from_beta = sin_beta / new_sin_ratio_from_beta

    print(" l_a {} \n l_b {} \n l_ab {} \n alpha {} \n beta  {} \n theta {} \n triangle {} \n phi {}".format(l_a, l_b, l_ab, deg(alpha), deg(beta), deg(theta), deg(alpha + beta + theta), deg(phi)))
    # for triangle-calculator.com
    # print("a={} b={} C={}".format(l_a, l_b, deg(theta)))
    print("\nmain triangle")
    print("https://www.triangle-calculator.com/?what=&q=a%3D{}+b%3D{}+C%3D{}&submit=Solve".format(l_a, l_b, deg(theta)))
    print("\ntriangle with new_length and b")
    print("https://www.triangle-calculator.com/?what=&q=b%3D{}+A%3D{}+C%3D{}&submit=Solve".format(l_b, deg(alpha), deg(phi)))
    print("\ntriangle with new_length and a")
    print("https://www.triangle-calculator.com/?what=&q=a%3D{}+B%3D{}+C%3D{}&submit=Solve".format(l_a, deg(beta), deg(theta_minus_phi)))
    print()
    assert float_equal(new_length, new_length_from_beta), "{} != {}".format(new_length, new_length_from_beta)

    result_vector = new_direction * new_length

    # plot their projection onto x-y plane because this should still work when replacing all z with 0
    p_0 = (0, 0)
    p_a = (a.vector[0], a.vector[1])
    p_b = (b.vector[0], b.vector[1])
    p_nd = tuple(new_direction[:2])
    p_rv = tuple(result_vector[:2])
    ps = [p_0, p_a, p_b, p_nd, p_rv]
    plt.scatter([p[0] for p in ps], [p[1] for p in ps], c="brrgk")
    lines = [[p_0, p_a], [p_0, p_b], [p_a, p_b], [p_0, p_rv]]
    lc = mc.LineCollection(lines)
    plt.gca().add_collection(lc)
    plt.show()

    return Moment(result_vector)

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def moment_multiply_array(arr):
    result = arr[0]
    for m in arr[1:]:
        result = moment_multiply(result, m)
    return result

def test_math():
    moments = [Moment(rand_unif_3d_sphere()) for _ in range(10)]
    z = Moment((0, 0, 0))
    moments2 = moments[:]
    random.shuffle(moments2)
    assert moments != moments2 and moments is not moments2

    #  identity, commutativity, and order of items
    for m in moments:
        assert m == moment_multiply(m, z) == moment_multiply(z, m)
        for m2 in moments2:
            mm2 = moment_multiply(m, m2)
            m2m = moment_multiply(m2, m)
            assert mm2 == m2m, "\n{}\n!=\n{}".format(mm2.vector, m2m.vector)
    assert moment_multiply_array(moments) == moment_multiply_array(moments2)

    # sin-like behavior
    m = Moment((1, 0, 0))
    thetas = np.arange(0, 2*np.pi, 0.01*np.pi)
    mags = []
    sins = []
    for theta in thetas:
        m2 = Moment((np.cos(theta), np.sin(theta), 0))
        prod = moment_multiply(m, m2)
        mag = prod.magnitude
        mags.append(mag)
        sins.append(np.sin(theta))
    plt.plot(thetas, mags, label="mags")
    plt.plot(thetas, sins, label="sins")
    plt.legend()
    plt.show()

    return True
    

if __name__ == "__main__":
    print(test_math())
    grains = get_grains(100)
    reference_point = (0, 0, 0)
    field = get_field_at_reference_point(grains, reference_point)
    print(field)
