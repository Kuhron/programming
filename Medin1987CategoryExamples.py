import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# seeing how MDS will place these items from p. 246
strs = ["1111", "1110", "1000", "0100", "0111", "0000", "0010", "1011", "0001", "1101"]
arr = np.array([[int(x) for x in s] for s in strs])
mds_fit = MDS().fit_transform(arr)
print(mds_fit)
xs, ys = mds_fit.T
plt.scatter(xs, ys)
for i in range(len(strs)):
    plt.gca().annotate(strs[i], (xs[i], ys[i]))

plt.show()

