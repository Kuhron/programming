import matplotlib.pyplot as plt
from datetime import datetime
import os



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
    def __init__(self, plot_every_n_steps=1, suppress_show=False, figsize=None):
        self.plot_every_n_steps = plot_every_n_steps
        self.suppress_show = suppress_show
        self.figsize = figsize

    def __enter__(self):
        self.counter = 0
        if not self.suppress_show:
            plt.ion()
            self.fignum = plt.gcf().number  # use to determine if user has closed plot
            self.fig = plt.gcf()  # failsafe if fignum gives false positive because it's always 1 when there's only one figure (e.g. if user closes and then a new figure is created before the next check)
        self.time_created = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # no exception occurred
            print("user closed plot; exiting")
            if not self.suppress_show:
                plt.ioff()
        else:
            if not self.suppress_show:
                plt.ioff()
            return False  # will reraise the exception

    def step(self, draw=True, savefig=False):
        self.counter += 1
        if self.should_plot():
            if draw:
                self.draw(savefig=savefig)
        else:
            return

    def draw(self, will_redraw=True, savefig=False):
        if not self.suppress_show:
            plt.draw()
            plt.pause(0.01)

        if savefig:
            fdir = "InteractivePlot-" + self.time_created
            fname = self.time_created + "-i" + str(self.counter) + ".png"
            if not os.path.exists(fdir):
                os.mkdir(fdir)
            fp = os.path.join(fdir, fname)
            if self.figsize is not None:
                plt.gcf().set_size_inches(*self.figsize)
            self.savefig(fp, dpi=100)

        if will_redraw:
            plt.gcf().clear()

    def force_draw_static(self, savefig=False):
        self.draw(will_redraw=False, savefig=savefig)

    def should_plot(self):
        return self.counter % self.plot_every_n_steps == 0

    def is_open(self):
        if self.suppress_show:
            return True
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
        return plt.plot(*args, **kwargs)

    @if_open
    def scatter(self, *args, **kwargs):
        return plt.scatter(*args, **kwargs)

    @if_open
    def contourf(self, *args, **kwargs):
        return plt.contourf(*args, **kwargs)

    @if_open
    def contour(self, *args, **kwargs):
        return plt.contour(*args, **kwargs)

    @if_open
    def colorbar(self, *args, **kwargs):
        return plt.colorbar(*args, **kwargs)

    @if_open
    def arrow(self, *args, **kwargs):
        return plt.arrow(*args, **kwargs)

    @if_open
    def subplot(self, *args, **kwargs):
        return plt.subplot(*args, **kwargs)

    @if_open
    def imshow(self, *args, **kwargs):
        return plt.imshow(*args, **kwargs)

    @if_open
    def title(self, *args, **kwargs):
        return plt.title(*args, **kwargs)

    def gcf(self, *args, **kwargs):
        return plt.gcf(*args, **kwargs)

    def gca(self, *args, **kwargs):
        return plt.gca(*args, **kwargs)

    @if_open
    def savefig(self, *args, **kwargs):
        print(f"saving fig at {args[0]}")
        return plt.savefig(*args, **kwargs)

