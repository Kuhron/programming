import matplotlib.pyplot as plt


class InteractivePlot:
    def __init__(self, plot_every_n_steps=1):
        self.plot_every_n_steps = plot_every_n_steps

    def __enter__(self):
        self.counter = 0
        plt.ion()
        self.fignum = plt.gcf().number  # use to determine if user has closed plot
        return self

    def __exit__(self, type, value, traceback):
        print("user closed plot; exiting")
        plt.ioff()

    def plot(self, *args, **kwargs):
        self.counter += 1
        if not self.should_plot():
            return

        plt.gcf().clear()
        plt.plot(*args, **kwargs)
        plt.draw()
        plt.pause(0.01)

    def should_plot(self):
        return self.counter % self.plot_every_n_steps == 0

    def is_open(self):
        return plt.fignum_exists(self.fignum)

    def is_closed(self):
        return not self.is_open()
