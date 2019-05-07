# try to get visualization for complex exponentiation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def update(slider_real, slider_imag, fig, ax):
    a = slider_real.val
    b = slider_imag.val
    z = a + b*1j
    # line.set_ydata(amp*np.sin(2*np.pi*freq*t))
    # line.set_ydata([x ** z for x in xs])
    ax.clear()
    plot_arrows(ax, z)
    fig.canvas.draw_idle()

def reset(slider_real, slider_imag):
    slider_real.reset()
    slider_imag.reset()

def colorfunc(label, fig):
    pass
    # line.set_color(label)
    # fig.canvas.draw_idle()

def get_complex_number_input():
    a = int(input("real part: "))
    b = int(input("imaginary part: "))
    return a + b*1j

def plot_arrows(ax, exponent):
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    rs = [0.5, 0.75, 1, 1.25, 1.5]
    n = 8
    thetas = 2 * np.pi * np.arange(0, 1, step=1/n)
    zs = [r * np.exp(1j * theta) for r in rs for theta in thetas]

    # zs = []
    # while len(zs) < 20:
    #     a = np.random.uniform(-2, 2)
    #     b = np.random.uniform(-2, 2)
    #     r = np.linalg.norm((a, b))
    #     if r <= 2:
    #         zs.append(a + b*1j)

    ys = [z ** exponent for z in zs]
    ax.scatter([z.real for z in zs], [z.imag for z in zs], c="b")
    ax.scatter([y.real for y in ys], [y.imag for y in ys], c="r")
    for z, y in zip(zs, ys):
        da = y.real - z.real
        db = y.imag - z.imag
        ax.arrow(z.real, z.imag, da, db)

    # plt.show()


if __name__ == "__main__":
    # z = get_complex_number_input()
    # plot_arrows(z)

    fig, ax = plt.subplots()
    # plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.subplots_adjust(bottom=0.25)
    # xs = np.arange(0.0, 1.0, 0.001)
    a0 = 1
    b0 = 0
    z0 = a0 + b0*1j
    delta_a = 0.01
    delta_b = 0.01
    # ys = [x ** z0 for x in xs]
    # line, *_ = plt.plot(xs, ys, color="red")
    # plt.axis([0, 1, -10, 10])
    plot_arrows(ax, z0)  # default plot

    axcolor = 'lightgoldenrodyellow'
    ax_real = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_imag = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    slider_real = Slider(ax_real, "Re(exponent)", -2, 2, valinit=a0, valstep=delta_a)
    slider_imag = Slider(ax_imag, "Im(exponent)", -2, 2, valinit=b0, valstep=delta_b)
    # sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
    # samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
    slider_real.on_changed(lambda *_: update(slider_real, slider_imag, fig, ax))
    slider_imag.on_changed(lambda *_: update(slider_real, slider_imag, fig, ax))

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    button.on_clicked(lambda *_: reset(slider_real, slider_imag))

    # rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    # radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
    # radio.on_clicked(colorfunc)

    plt.show()
