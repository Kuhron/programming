import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def get_single_trajectory(
        n_steps,
        starting_value,
        control_points,
        mus,
        sigmas,
    ):
    
    x = starting_value
    xs = [x]
    mu_function = get_mu_function(control_points, mus)
    sigma_function = get_sigma_function(control_points, sigmas)
    for i in range(n_steps-1):
        mu = mu_function(x)
        sigma = sigma_function(x)
        change = random.normalvariate(mu, sigma)
        x += change
        xs.append(x)
    return xs


def get_mu_function(control_points, mus):
    control_points = control_points + [control_points[-1] + 1]
    mus = mus + [mus[-1]]
    mu_interp = interp1d(control_points, mus, fill_value="extrapolate")
    mu = lambda x: np.sign(x) * mu_interp(abs(x))
    return mu


def get_sigma_function(control_points, sigmas):
    control_points = control_points + [control_points[-1] + 1]
    sigmas = sigmas + [sigmas[-1]]
    sigma_interp = interp1d(control_points, sigmas, fill_value="extrapolate")
    sigma = lambda x: sigma_interp(abs(x))
    return sigma


def plot_mus_and_sigmas(max_abs, control_points, mus, sigmas):
    xs = np.linspace(-max_abs, max_abs, 1000)
    mu = get_mu_function(control_points, mus)
    sigma = get_sigma_function(control_points, sigmas)
    plt.plot(xs, mu(xs), c="b", label="mu")
    plt.plot(xs, sigma(xs), c="r", label="sigma")
    plt.legend()
    plt.show()


def plot_multiple_trajectories(n_trajectories, n_steps, *args, **kwargs):
    trs = []
    for tr_i in range(n_trajectories):
        trs.append(get_single_trajectory(n_steps, *args, **kwargs))
    xs = list(range(n_steps))
    for tr in trs:
        plt.plot(xs, tr, c="k", alpha=0.1)
    plt.show()


if __name__ == "__main__":
    plot_mus_and_sigmas(max_abs=1500,
        control_points = [0, 25, 100, 1000],
        mus = [0, -1, 10, 0],
        sigmas = [10, 15, 30, 100],
    )
    plot_multiple_trajectories(
        n_trajectories = 10,
        n_steps = 1000,
        starting_value = 0, 
        control_points = [0, 25, 100, 1000],
        mus = [0, -1, 10, 0],
        sigmas = [10, 15, 30, 100],
    )
 
