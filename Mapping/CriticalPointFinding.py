raise Exception("try brute force search first before worrying about this")
# TODO to make this do anything more than glorified brute-force, 
# need to incorporate knowledge about the secant line slopes 
# to restrict how far away y can go within each interval

# # finding critical point (local max/min) in trajectory of distances from a point to an edge
# # assuming there will be at most one


# import numpy as np
# import matplotlib.pyplot as plt


# def find_extrema_in_interval(f, x0, x1, x_tolerance, midpoint_func):
#     # assumes f has at most one critical point on this interval
#     # if there are none, then the extrema must be the endpoints
#     xm = midpoint_func(x0, x1)
#     y0 = f(x0)
#     y1 = f(x1)
#     ym = f(xm)
#     if abs(x1 - x0) < x_tolerance:
#         ymin = ym
#         ymax = ym    
#     elif y0 == ym == y1:
#         # case 111 (all three are equal)
#         # critical point is not possible
#         ymin = ym
#         ymax = ym
#     elif y0 == ym:
#         # cases 112 and 221 (two one one side are equal, the third is not)
#         # critical point must be in between the equal ones (by MVT there must be one)
#         ymin, ymax = find_extrema_in_interval(f, x0, xm, x_tolerance, midpoint_func)
#     elif ym == y1:
#         # cases 122 and 211 (horizontal mirror of 112 and 221)
#         ymin, ymax = find_extrema_in_interval(f, xm, x1, x_tolerance, midpoint_func)
#     elif (y0 < ym > y1) or (y0 > ym < y1):
#         # cases 121, 212, 132, 213, 231, 312 (middle is larger/smaller than both of the outside points)
#         # critical point could be in left half, right half, or right in the middle
#         # by MVT there must be one
#         yleft_min, yleft_max = find_extrema_in_interval(f, x0, xm, x_tolerance, midpoint_func)
#         yright_min, yright_max = find_extrema_in_interval(f, xm, x1, x_tolerance, midpoint_func)
#         ymin = min(yleft_min, yright_min, ym)
#         ymax = max(yleft_max, yright_max, ym)
#     elif (y0 < ym < y1) or (y0 > ym > y1):
#         # cases 123 and 321 (middle is between the outside points)
#         # critical point could be on one endpoint or in either interval, but can't be the middle
#         # there could also be no critical point
#         raise NotImplementedError
#     else:
#         raise ValueError("unknown situation")
    
#     return ymin, ymax
    

#     raise NotImplementedError


# if __name__ == "__main__":
#     x_tolerance = 0.05
#     while True:
#         a,b,c = np.random.normal(0, 3, (3,))
#         f = lambda x: a*x**2 + b*x + c
#         xs = np.linspace(0, 1, 100)
#         ys = f(xs)
#         extrema = find_extrema_in_interval(f, xs[0], xs[-1], x_tolerance)
#         print(extrema)
#         plt.plot(xs, ys)
#         plt.show()
