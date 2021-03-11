import random
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl


# penny = direction: heads = beneficial, tails = detrimental
# nickel = domain/realm: heads = mental, tails = physical
# dime = dynamism: heads = dynamic, tails = static
# quarter = scope: heads = individual, tails = collective

# each coin has a "heading", a direction it points
# if there is a face (including the eagle), the heading is the direction the face is looking, if it has a predominant component parallel to the ground
# if the face is only looking perpendicular to the ground (e.g. straight at the viewer), or if there is no face, the heading is upward (toward the top of the design)


penny_width_mm = 19.5
nickel_width_mm = 21.21
dime_width_mm = 17.9
quarter_width_mm = 24.26
coin_widths = {
    "p": penny_width_mm,
    "n": nickel_width_mm,
    "d": dime_width_mm,
    "q": quarter_width_mm,
}
coin_colors = {
    "p": "darkgoldenrod",
    "n": "silver",
    "d": "cadetblue",
    "q": "dimgrey",
}
coin_meanings = {
    "p": ["beneficial", "detrimental"],
    "n": ["mental", "physical"],
    "d": ["dynamic", "static"],
    "q": ["individual", "collective"],
}


def throw_coins():
    coins = "pndq"
    fall_order = random.sample(coins, len(coins))
    # fall order has the first thing touching the ground earliest, so the first one in fall order will be lowest in the case of overlapping coins
    # assume 2d normal distribution of positions
    # the space should be big enough for the coins to fall fairly far apart but also reasonably likely to overlap
    sigma_mm = quarter_width_mm * 3
    xs, ys = np.random.normal(0, sigma_mm, (2, len(coins)))
    headings = np.random.uniform(0, 2*np.pi, (len(coins),))

    for coin, x, y, heading in zip(fall_order, xs, ys, headings):
        width = coin_widths[coin]
        color = coin_colors[coin]
        circle_patch = plt.Circle((x,y), radius=width, alpha=1, edgecolor="black", facecolor=color)
        plt.gca().add_patch(circle_patch)
        heads_tails = random.choice([0,1])
        annotation = coin + "HT"[heads_tails] + " = " + coin_meanings[coin][heads_tails]

        arrow_dx = width * 2 * np.cos(heading)
        arrow_dy = width * 2 * np.sin(heading)
        plt.arrow(x, y, arrow_dx, arrow_dy, edgecolor="black", facecolor=color, head_width=0.5*dime_width_mm)
        plt.annotate(annotation, (x + width * 1.2, y))

    xlim = max(abs(xs)) + quarter_width_mm * 1.5
    ylim = max(abs(ys)) + quarter_width_mm * 1.5
    lim = max(xlim, ylim)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect('equal')

    plt.show()


if __name__ == "__main__":
    throw_coins()
