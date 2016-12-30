import datetime
import math
import random
import signal
import time
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import Exchange.Contracts as contracts
from FileFollower import follow
from WorkerListener import Worker, Listener


class TimeSeriesWriter(Listener):
    def __init__(self, queue, underlying):
        super().__init__(queue)

        contracts.verify_underlying(underlying)
        self.underlying = underlying
        self.filepath = contracts.get_filepath_from_underlying(self.underlying)

    def process_item(self, item):
        with open(self.filepath, "a") as f:
            now = time.time()
            f.write("{:.6f},{:.4f}\n".format(now, item))
        print("wrote", item)


class TimeSeriesReader(Worker):
    def __init__(self, queue, underlying):
        super().__init__(queue)
        
        self.underlying = underlying
        self.filepath = contracts.get_filepath_from_underlying(self.underlying)

    def run(self):
        for item in follow(self.filepath):
            self.put_item(item)


class TimeSeriesPlotter(Listener):
    def __init__(self, queue, underlying):
        super().__init__(queue)

        self.underlying = underlying
        self.xs = []
        self.ys = []
        self.start_plot()
        self.last_redraw_time = None
        self.redraw_delay_seconds = 0.1
        self.n_points = 100

    def start_plot(self):
        print("starting plot")
        plt.ion()
        # plt.show()  # shouldn't be necessary but somehow the plot is not showing?

    def ready_to_redraw(self):
        return (self.last_redraw_time is None) or (time.time() - self.last_redraw_time > self.redraw_delay_seconds)

    def redraw(self):
        # plt.gca().clear()
        plt.clf()
        plt.plot(self.xs, self.ys)
        plt.pause(0.01)

        self.last_redraw_time = time.time()

    def process_item(self, item):
        print("got item", item)
        line = item
        t, val = contracts.parse_line(line)

        self.xs.append(t)
        self.ys.append(val)
        self.xs = self.xs[-self.n_points:]
        self.ys = self.ys[-self.n_points:]
        if self.ready_to_redraw():
            print("attempting redraw")
            self.redraw()
        else:
            print("not redrawing at this time")


class TimeSeriesCreator(Worker):
    def __init__(self, queue, underlying):
        super().__init__(queue)

        self.underlying = underlying
        self.last_value = None
        self.delay_seconds = 1

    def run(self):
        if contracts.underlying_has_data(self.underlying):
            self.last_value = contracts.get_spot(self.underlying)
        else:
            self.last_value = math.exp(random.normalvariate(6, 3))

        while True:
            sd = abs(random.normalvariate(0, 0.02))
            factor = math.exp(random.normalvariate(0, sd))
            self.last_value *= factor
            self.put_item(self.last_value)
            time.sleep(self.delay_seconds)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def write_ts(underlying):
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(2, initializer=init_worker, initargs=())

    listener = TimeSeriesWriter(queue, underlying)
    pool_listener = pool.apply_async(listener.run, ())

    worker = TimeSeriesCreator(queue, underlying)
    worker_args = ()
    job = pool.apply_async(worker.run, worker_args)
    job.get()

    queue.put(StopIteration)
    pool.close()
    pool.join()


def plot_ts_continuous(underlying):
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(2, initializer=init_worker, initargs=())

    listener = TimeSeriesPlotter(queue, underlying)
    pool_listener = pool.apply_async(listener.run, ())

    worker = TimeSeriesReader(queue, underlying)
    worker_args = ()
    job = pool.apply_async(worker.run, worker_args)
    job.get()

    queue.put(StopIteration)
    pool.close()
    pool.join()


def plot_ts_static(underlying):
    xs = []
    ys = []

    with open(contracts.get_filepath_from_underlying(underlying)) as f:
        lines = f.readlines()

    for line in lines:
        t, val = contracts.parse_line(line)
        xs.append(t)
        ys.append(val)

    plt.plot(xs, ys)
    plt.show()


if __name__ == "__main__":
    u = input("underlying: ")
    contracts.verify_underlying(u)

    a = input("1. write\n2. plot static\n3. plot continuous (not working)\n")
    if a == "1":
        write_ts(u)
    elif a == "2":
        plot_ts_static(u)
    elif a == "3":
        plot_ts_continuous(u)
    else:
        raise