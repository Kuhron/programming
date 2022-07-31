import matplotlib.pyplot as plt



class PlotClosedError(BaseException):
    pass


def if_open(func):
    # the "iplt" arg to the wrapper is the "self" from within the class instance
    def new_func(iplt, *args, **kwargs):
        if iplt.is_open():
            func(iplt, *args, **kwargs)
        else:
            print("iplt is closed")
            raise PlotClosedError
    return new_func


class InteractivePlot():
    def __init__(self, plot_every_n_steps=1):
        self.plot_every_n_steps = plot_every_n_steps

    def __enter__(self):
        self.counter = 0
        plt.ion()
        self.fignum = plt.gcf().number  # use to determine if user has closed plot
        self.fig = plt.gcf()  # failsafe if fignum gives false positive because it's always 1 when there's only one figure (e.g. if user closes and then a new figure is created before the next check)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # no exception occurred
            print("user closed plot; exiting")
            plt.ioff()
        else:
            plt.ioff()
            return False  # will reraise the exception

    def step(self):
        self.counter += 1
        if self.should_plot():
            self.draw()
        else:
            return

    def draw(self, will_redraw=True):
        plt.draw()
        plt.pause(0.01)
        if will_redraw:
            plt.gcf().clear()

    def force_draw_static(self):
        self.draw(will_redraw=False)

    def should_plot(self):
        return self.counter % self.plot_every_n_steps == 0

    def is_open(self):
        return plt.fignum_exists(self.fignum) and self.fig_is_same()

    def is_closed(self):
        return not self.is_open()

    def fig_is_same(self):
        return self.fig is plt.gcf()


    # these are just "inheriting" from plt
    # (which is a module, not a class, so you can't inherit from it)
    # if I can figure out how to get this to work in general that would be great
    # using getattr/hasattr causes infinite recursion

    @if_open
    def plot(self, *args, **kwargs):
        plt.plot(*args, **kwargs)

    @if_open
    def contourf(self, *args, **kwargs):
        plt.contourf(*args, **kwargs)

    @if_open
    def contour(self, *args, **kwargs):
        plt.contour(*args, **kwargs)

    @if_open
    def colorbar(self, *args, **kwargs):
        plt.colorbar(*args, **kwargs)

    @if_open
    def arrow(self, *args, **kwargs):
        plt.arrow(*args, **kwargs)

    @if_open
    def subplot(self, *args, **kwargs):
        plt.subplot(*args, **kwargs)

    @if_open
    def imshow(self, *args, **kwargs):
        plt.imshow(*args, **kwargs)

    @if_open
    def title(self, *args, **kwargs):
        plt.title(*args, **kwargs)


