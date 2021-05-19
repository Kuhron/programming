# inspired by Carr et al. 2017 study of emergent language to describe triangles
# try making my own similarity judgments and doing MDS on them like they did in the paper

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os


class LineSegmentStimulus:
    def __init__(self, xys, designation):
        assert xys.shape == (4,2)
        self.xys = xys
        assert type(designation) is str
        self.designation = designation

    def plot(self):
        xs = self.xys[:,0]
        ys = self.xys[:,1]
        plt.plot(xs, ys)

    @staticmethod
    def random(designation):
        xys = np.random.randint(0, 101, (4,2))
        return LineSegmentStimulus(xys, designation)

    def __repr__(self):
        s = "<LineSegmentStimulus:"
        s += f"{self.designation}:"
        for row_i in range(4):
            x,y = self.xys[row_i,:]
            s += f"{x},{y};"
        s += ">"
        return s

    @staticmethod
    def from_str(s):
        beginning = "<LineSegmentStimulus:"
        assert s.startswith(beginning)
        ending = ">"
        assert s.endswith(ending)
        s = s.replace(beginning, "").replace(ending, "")
        designation, s = s.split(":")
        xy_strs = s.split(";")
        assert xy_strs[-1] == ""
        assert len(xy_strs) == 5
        xys = [[float(x) for x in xy.split(",")] for xy in xy_strs[:-1]]
        xys = np.array(xys)
        return LineSegmentStimulus(xys, designation)


class Response:
    def __init__(self, base_stimulus, option_a, option_b, answer):
        self.base_stimulus = base_stimulus
        self.option_a = option_a
        self.option_b = option_b
        self.answer = answer

    def to_record_str(self):
        return f"{self.base_stimulus.designation};;{self.option_a.designation};;{self.option_b.designation};;{self.answer}"


def plot_stimuli(base_stimulus, option_a, option_b):
    plt.subplot(2,2,1)
    base_stimulus.plot()
    plt.title("baseline")

    func_scope_lst = []  # https://stackoverflow.com/questions/15032638/how-to-return-a-value-from-button-press-event-matplotlib

    def select_callback(option):
        plt.close()
        response = Response(base_stimulus, option_a, option_b, answer=option)
        func_scope_lst.append(response)

    plt.subplot(2,2,3)
    option_a.plot()
    plt.title("option a")
    button_a = Button(plt.gca(), "Option A is more similar")
    button_a.on_clicked(lambda event: select_callback("a"))

    plt.subplot(2,2,4)
    option_b.plot()
    plt.title("option b")
    button_b = Button(plt.gca(), "Option B is more similar")
    button_b.on_clicked(lambda event: select_callback("b"))

    plt.show()

    if len(func_scope_lst) == 0:
        return None
    else:
        assert len(func_scope_lst) == 1
        return func_scope_lst[0]


def append_response(response, record_fp):
    with open(record_fp, "a") as f:
        f.write(response.to_record_str() + "\n")


def create_static_stimuli(fp):
    stimuli = [LineSegmentStimulus.random(designation=str(i)) for i in range(2000)]
    with open(fp, "w") as f:
        for stimulus in stimuli:
            f.write(f"{stimulus}\n")


def get_static_stimuli(static_stimuli_fp):
    with open(static_stimuli_fp) as f:
        lines = f.readlines()
    res = []
    for line in lines:
        line = line.strip()
        stimulus = LineSegmentStimulus.from_str(line)
        res.append(stimulus)
    return res


def record_similarity_judgments(static_stimuli_fp, record_fp):
    stimuli = get_static_stimuli(static_stimuli_fp)
    while True:
        base_stimulus, option_a, option_b = random.sample(stimuli, 3)
        response = plot_stimuli(base_stimulus, option_a, option_b)
        answer = response.answer
        if answer not in ["a", "b"]:
            print("invalid answer; skipping")
            continue
        print(f"you chose {answer}")
        append_response(response, record_fp)


if __name__ == "__main__":
    record_fp = "SubjectiveSimilarityJudgments.txt"
    static_stimuli_fp = "SubjectiveSimilarityStaticStimuli.txt"

    if not os.path.exists(record_fp):
        open(record_fp, "w").close()
    if not os.path.exists(static_stimuli_fp):
        create_static_stimuli(static_stimuli_fp)

    record_similarity_judgments(static_stimuli_fp, record_fp)
