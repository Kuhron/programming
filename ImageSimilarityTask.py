# have the user do some choice, ranking, or rating task to get data about how similar they consider certain pairs of images to be
# simplest would be show them one image, and have them choose which of two others is most similar to the first one
# could also do something like odd one out, show them three images and have them choose the one that is the least like the other two
# or rank N images in terms of similarity to a reference image
# can play with giving user different tasks like this, store data on the answers
# see how the images can be placed into a space based on similarity, like MDS from precomputed distance matrix, see what clusters arise

# to convert odd-one-out data to similarity scores, could count something like number of times each pair has been voted as the similar one out of the three, and how many trials, e.g. if user picks 53 as odd one out among [24, 53, 88], then [24, 88] gets 1 success out of 1 trial, [24, 53] and [53, 88] each get 0 successes out of 1 trial. Try this and see how well it gives you numbers. Hopefully can find a way to not try to impute for unobserved pairs since I don't at all trust that these ratings will be transitive.


import random
import itertools
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import networkx as nx

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QPushButton, QLabel

from BinomialObservation import BinomialObservation


def select_image_index(i_s, selected_i):
    i0, i1, i2 = i_s
    if selected_i == i0:
        return [i0, i1, i2]
    elif selected_i == i1:
        return [i1, i0, i2]
    elif selected_i == i2:
        return [i2, i0, i1]
    else:
        raise ValueError(f"{selected_i} not in {[i0, i1, i2]}")


def select_image(i_s, selected_i, odd_one_out_data, buttons, fps, output_fp, missing_pairs):
    print(f"selected image {selected_i} from {i_s}")
    lst = select_image_index(i_s, selected_i)
    odd_one_out_data.append(lst)
    with open(output_fp, "a") as f:
        while len(odd_one_out_data) > 0:
            s = " ".join(str(x) for x in odd_one_out_data[0])
            print(s)
            f.write(s + "\n")
            odd_one_out_data.remove(odd_one_out_data[0])
    populate_buttons(buttons, fps, output_fp, missing_pairs)


def populate_buttons(buttons, fps, output_fp, missing_pairs=None):
    if missing_pairs is None or len(missing_pairs) == 0:
        i0, i1, i2 = random.sample(range(len(fps)), 3)
    else:
        # prioritize one of the pairs that is missing
        i0, i1 = random.choice(list(missing_pairs))
        i2 = random.choice([i for i in range(len(fps)) if i != i0 and i != i1])
    missing_pairs -= {tuple(sorted([i0, i1])), tuple(sorted([i0, i2])), tuple(sorted([i1, i2]))}
    print(f"now have {len(missing_pairs)} missing pairs")

    i_s = [i0, i1, i2]
    for i, img_i in enumerate(i_s):
        img = fps[img_i]
        icon = QIcon(img)
        buttons[i].setIcon(icon)
        try:
            buttons[i].pressed.disconnect()
        except:
            print("nothing to disconnect")
        buttons[i].pressed.connect(lambda selected_i=i_s[i]: select_image(i_s, selected_i, odd_one_out_data, buttons, fps, output_fp, missing_pairs))


def get_random_topological_sort(g):
    # I don't know if this is biased, it probably is, but I just want to see how well I can get proxy edge weights by doing this
    candidates_for_next = [n for n in g.nodes if g.in_degree(n) == 0]
    res = []
    while len(res) < len(g.nodes):
        n = random.choice(candidates_for_next)
        res.append(n)
        out_edges = g.out_edges([n], data=False)



if __name__ == "__main__":
    run = False
    image_dir = "Emergence/Images/YellowBlueBlobRegion_50/"
    output_fp = os.path.join(image_dir, "ImageSimilarityData.txt")
    fnames = [x for x in os.listdir(image_dir) if x.endswith(".png")]
    fps = [os.path.join(image_dir, x) for x in fnames]
    n_images = len(fps)

    if run:
        app = QApplication(sys.argv)
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        label = QLabel("Select the odd one out")
        layout.addWidget(label)
        odd_one_out_data = [["odd_one_out", "similar1", "similar2"]]  # keep this so it puts this line in the txt file at the start of each session, can easily filter them out, might help to know which ratings occurred in the same session (e.g. in case file index numbers get scrambled)
        buttons = []

        for i in range(3):
            button = QPushButton()
            layout.addWidget(button)
            buttons.append(button)
            button.setIconSize(QSize(256, 256))

        with open(output_fp) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines if not x.startswith("odd")]
        l = [[int(x) for x in s.split(" ")] for s in lines]
        missing_pairs = {(i,j) for i in range(n_images) for j in range(i+1, n_images)}
        for a,b,c in l:
            missing_pairs -= {tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))}

        populate_buttons(buttons, fps, output_fp, missing_pairs)
        widget.show()
        app.exec_()

    else:
        # analyze
        with open(output_fp) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines if not x.startswith("odd")]
        l = [[int(x) for x in s.split(" ")] for s in lines]
        difference_ratings = {}
        trials = {}
        for a,b,c in l:
            ab = tuple(sorted([a,b]))
            ac = tuple(sorted([a,c]))
            bc = tuple(sorted([b,c]))
            if ab not in difference_ratings:
                assert ab not in trials
                difference_ratings[ab] = 0
                trials[ab] = 0
            if ac not in difference_ratings:
                assert ac not in trials
                difference_ratings[ac] = 0
                trials[ac] = 0
            if bc not in difference_ratings:
                assert bc not in trials
                difference_ratings[bc] = 0
                trials[bc] = 0
            # a was rated as more different from b and c, b and c were rated as more similar to each other
            difference_ratings[ab] += 1
            difference_ratings[ac] += 1
            # do not increment bc, the most similar pair
            trials[ab] += 1
            trials[ac] += 1
            trials[bc] += 1
        distances = [[0 if i == j else None for j in range(n_images)] for i in range(n_images)]
        for i, j in difference_ratings.keys():
            assert 0 <= i < n_images, i
            assert 0 <= j < n_images, j
            if i != j:
                assert (j,i) not in difference_ratings
                assert (j,i) not in trials
            s = difference_ratings[(i,j)]
            t = trials[(i,j)]
            b = BinomialObservation(s, t)
            d = b.get_centered_wilson_estimator(0)
            distances[i][j] = d
            distances[j][i] = d
        # print("\n".join(" ".join("-" if distances[i][j] is None else "X" if distances[i][j] == 1 else str(int(distances[i][j]/0.1)) for j in range(n_images)) for i in range(n_images)))
        distances = np.array(distances).astype(float)
        # plt.imshow(distances)
        # plt.colorbar()
        # plt.show()

        # mds = MDS(dissimilarity="precomputed")
        # # non-metric (where distances just show rank, not quantitative distance) seems to give really unstable results, qualitatively different clusters on each run
        # X = mds.fit_transform(distances)
        # plt.scatter(X[:,0], X[:,1])
        # for i in range(n_images):
        #     x, y = X[i]
        #     plt.annotate(str(i), (x, y))
        # plt.show()

        g = nx.DiGraph()
        pairs = [tuple(sorted(x)) for x in itertools.combinations(range(n_images), 2)]
        for pair in pairs:
            g.add_node(pair)  # make sure every edge has a node so the topological sort will include all of them
        relative_ordering_successes = {}  # number of times the first edge is LONGER than the second edge
        relative_ordering_trials = {}
        # if there's an X% chance that one pair is farther apart than another pair, add an arrow from the closer pair to the farther pair, else, remain agnostic about ordering of those edges' weights
        # then do topological sort to get a possible ordering of the edge weights, then distribute edge weights uniformly on [0,1] to reflect that ordering, record the distance value for each edge made this way, do it some more times, average the weights you got for each edge, then use THESE as the distance matrix for MDS
        # this has advantage that we don't need to have seen all pairs of images, some of these edges will just have no observations and so the digraph ordering the edge weights will be totally agnostic about arrows to/from that edge, so a topological sort giving an edge weight ordering will still be possible albeit less accurate
        for a,b,c in l:
            # this tells us that, according to the user at this time, the edge ab and ac are both longer than bc, says nothing about relative ordering of ab and ac
            ab = tuple(sorted([a,b]))
            ac = tuple(sorted([a,c]))
            bc = tuple(sorted([b,c]))
            ab_bc = tuple(sorted([ab,bc]))
            ac_bc = tuple(sorted([ac,bc]))
            for epair in [ab_bc, ac_bc]:
                if epair not in relative_ordering_successes:
                    assert epair not in relative_ordering_trials
                    relative_ordering_successes[epair] = 0
                    relative_ordering_trials[epair] = 0
                # what we know here is that one of the edges in epair is bc, and the other edge is longer than it
                e1, e2 = epair
                if e1 == bc:
                    # e1 < e2, so first edge is not longer than second edge, failure
                    pass  # don't increment successes
                elif e2 == bc:
                    # e1 > e2, success
                    relative_ordering_successes[epair] += 1
                else:
                    raise Exception("bc should be in epair")
                relative_ordering_trials[epair] += 1
        for epair in relative_ordering_successes.keys():
            e1, e2 = epair
            s = relative_ordering_successes[epair]
            t = relative_ordering_trials[epair]
            b = BinomialObservation(s, t)
            lower, upper = b.get_wilson_ci(0.01)
            if upper < 0.5:
                # confident that this pair fails, meaning e1 is shorter than e2
                # arrow in graph should go from shorter edge to longer edge, since things at the ends of arrows will be later in the topological sort
                # so put arrow from e1 to e2
                # assuming the DiGraph edge is (arrow_from, arrow_to), I can't find this explicitly stated in nx docs!
                g.add_edge(e1, e2)
            elif lower > 0.5:
                # confident that this pair succeeds, meaning that e1 is longer than e2
                # so put arrow from e2 to e1
                g.add_edge(e2, e1)
            else:
                # not confident about relative weights of these edges
                pass
        print(f"added arrow for {len(g.edges)/len(relative_ordering_successes.keys()):.2%} of edge pairs which have any ordering observations, for {len(g.edges)/(lambda n: n*(n-1)/2)(len(pairs)):.2%} of all edge pairs")

        # now get some topological sorts of the graph to sort of sample what the edge weights (i.e., the distances between some images) could be
        # but there are way too many possible topological sorts
        edge_weight_observations = {pair: [] for pair in pairs}
        for i in range(100):
            print(f"topological sort #{i}")
            # tsort = list(nx.topological_sort(g))
            tsort = get_random_topological_sort(g)
            min_weight = 0
            max_weight = 1
            for j, e in enumerate(tsort):
                # 0th edge has shortest weight, =min_weight
                # last edge (j=len(tsort) - 1) has longest weight, =max_weight
                weight = min_weight + (max_weight - min_weight)/(len(tsort)-1 - 0)
                if e not in edge_weight_observations:
                    edge_weight_observations[e] = []
                edge_weight_observations[e].append(weight)
        print(edge_weight_observations)
