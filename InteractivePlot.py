import matplotlib.pyplot as plt


class InteractivePlot():
    def __init__(self, plot_every_n_steps=1):
        self.plot_every_n_steps = plot_every_n_steps

    def __enter__(self):
        self.counter = 0
        plt.ion()
        self.fignum = plt.gcf().number  # use to determine if user has closed plot
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # no exception occurred
            print("user closed plot; exiting")
            plt.ioff()
        else:
            plt.ioff()
            return False  # will reraise the exception

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


    # these are just "inheriting" from plt
    # (which is a module, not a class, so you can't inherit from it)
    # if I can figure out how to get this to work in general that would be great
    # using getattr/hasattr causes infinite recursion

    def contourf(self, *args, **kwargs):
        plt.contourf(*args, **kwargs)

    def contour(self, *args, **kwargs):
        plt.contour(*args, **kwargs)

    def colorbar(self, *args, **kwargs):
        plt.colorbar(*args, **kwargs)

    def arrow(self, *args, **kwargs):
        plt.arrow(*args, **kwargs)


