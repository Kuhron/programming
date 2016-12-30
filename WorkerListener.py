# copied from http://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
# refactored into classes
# SIGINT handling copied from second answer to http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool


import multiprocessing as mp
import signal
import time

import random


class Worker:
    def __init__(self, queue):
        self.queue = queue

    def put_item(self, item):
        self.queue.put(item)

    def run(self):
        NotImplemented


class Listener:
    def __init__(self, queue):
        self.queue = queue

    def get_item(self):
        return self.queue.get()

    def has_item(self):
        return not self.queue.empty()

    def run(self):
        while True:
            if self.has_item():
                item = self.get_item()
                # print("got item:", item)

                if item is StopIteration:
                    return

                self.process_item(item)
            else:
                time.sleep(0.1)

    def process_item(self, item):
        NotImplemented


class TestWorker(Worker):
    def run(self, n):
        time.sleep(n)
        result = random.uniform(-n, n)
        self.put_item(result)


class TestContinualWorker(Worker):
    def run(self, n):
        val = 0
        while True:
            time.sleep(0.1)
            change = random.uniform(-n, n)
            val += change
            self.put_item(val)


class TestListener(Listener):
    def process_item(self, item):
        s = "{:.4f}".format(item)
        with open("a.txt", "a") as f:
            f.write(s + "\n")


def test_many_workers():
    # must use Manager queue here, or will not work
    manager = mp.Manager()
    queue = manager.Queue()
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool()
    signal.signal(signal.SIGINT, original_sigint_handler)

    try:
        listener = TestListener(queue)

        # put listener to work first
        pool_listener = pool.apply_async(listener.run, ())

        # fire off workers
        jobs = []
        for i in range(100):
            worker = TestWorker(queue)
            worker_args = (i, )
            job = pool.apply_async(worker.run, worker_args)  # returns AsyncResult object
            jobs.append(job)

        # collect results from the workers through the pool result queue
        for job in jobs:
            job.get()  # return the result when it arrives, unless timeout seconds pass
            # note that TestWorker illustrates that run() does not have to return anything; it just puts things on the queue

        # now we are done, close the queue to indicate no more data will be sent
        queue.put(StopIteration)
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        queue.put(StopIteration)
        pool.terminate()

    else:
        queue.put(StopIteration)
        pool.close()

    pool.join()


def test_continual_worker():
    # must use Manager queue here, or will not work
    manager = mp.Manager()
    queue = manager.Queue()
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool()
    signal.signal(signal.SIGINT, original_sigint_handler)

    try:
        listener = TestListener(queue)

        # put listener to work first
        pool_listener = pool.apply_async(listener.run, ())

        # fire off workers
        jobs = []
        worker = TestContinualWorker(queue)
        worker_args = (100, )
        job = pool.apply_async(worker.run, worker_args)  # returns AsyncResult object
        jobs.append(job)

        # collect results from the workers through the pool result queue
        for job in jobs:
            job.get()  # return the result when it arrives, unless timeout seconds pass

        # now we are done, close the queue to indicate no more data will be sent
        queue.put(StopIteration)
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        queue.put(StopIteration)
        pool.terminate()

    else:
        queue.put(StopIteration)
        pool.close()

    pool.join()


if __name__ == "__main__":
    # test_many_workers()
    test_continual_worker()