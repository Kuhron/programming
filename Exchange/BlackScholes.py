# copied from http://janroman.dhis.org/stud/I2014/BS2/BS_Daniel.pdf
# cleaned up a bit, left a lot as is because who cares

"""
# The Black Scholes Formula
# CallPutFlag - This is set to 'c' for call option, anything else for put
# S - Stock price
# K - Strike price
# T - Time to maturity
# r - Riskfree interest rate
# d - Dividend yield
# v - Volatility
"""

from scipy.stats import norm
from math import exp, log, sqrt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def animate_price_to_expiry(call_put_flag,S,K,T,r,d,v):
    fig,ax = plt.subplots()
    # maturity = 0
    S_list = np.linspace(0.8*S,1.2*S,200)
    p = []
    for S in S_list:
        p.append(black_scholes(call_put_flag, S, K, T, r, d, v)["price"])
        line, = ax.plot(p)

    def update(step):
        p = []
        for S in S_list:
            p.append(black_scholes(call_put_flag, S, K, step, r, d, v)["price"])
            line.set_ydata(p)

    def data_gen():
        expStop = 0.0005
        expStart = 1.5
        T = np.linspace(expStop,expStart,200)
        m =-log(expStop/expStart)/expStart
        for t in T:
            yield expStart*exp(-m*t)

    ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
    plt.show()


def black_scholes_greeks_call(S, K, T, r, d, v):
    T_sqrt = sqrt(T)
    d1 = (log(float(S)/K)+((r-d)+v*v/2.)*T)/(v*T_sqrt)
    d2 = d1-v*T_sqrt
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    ert = exp(-r*T)
    price = S*exp(-d*T)*Nd1-K*ert*Nd2
    Delta = Nd1
    Gamma = nd1/(S*v*T_sqrt)
    Theta =-(S*v*nd1)/(2*T_sqrt)-r*K*ert*Nd2
    Vega = S * T_sqrt*nd1
    Rho = K*T*ert*Nd2
    return {
        "price": price,
        "delta": Delta,
        "gamma": Gamma,
        "theta": Theta,
        "vega": Vega,
        "rho": Rho,
    }


def black_scholes_greeks_put(S, K, T, r, d, v):
    T_sqrt = sqrt(T)
    # d1 = (log(float(S)/K)+r*T)/(v*T_sqrt) + 0.5*v*T_sqrt  # original; bug?
    d1 = (log(float(S)/K)+((r-d)+v*v/2.)*T)/(v*T_sqrt)
    d2 = d1-(v*T_sqrt)
    nd1 = norm.pdf(d1)
    N_d1 = norm.cdf(-d1)
    N_d2 = norm.cdf(-d2)
    ert = exp(-r*T)
    price = K*ert*N_d2-S*exp(-d*T)*N_d1
    Delta = -N_d1
    Gamma = nd1/(S*v*T_sqrt)
    Theta =-(S*v*nd1) / (2*T_sqrt)+ r*K * ert * N_d2
    Vega  = S * T_sqrt * nd1
    Rho   =-K*T*ert * N_d2
    return {
        "price": price,
        "delta": Delta,
        "gamma": Gamma,
        "theta": Theta,
        "vega": Vega,
        "rho": Rho,
    }


def black_scholes(call_put_flag, S, K, T, r, d, v):
    assert call_put_flag in ["c", "p"]
    if call_put_flag == "c":
        return black_scholes_greeks_call(S, K, T, r, d, v)
    else:
        return black_scholes_greeks_put(S, K, T, r, d, v)


if __name__ == "__main__":
    S = 110
    K = 100
    T = 0.05
    r = 0
    d = 0
    v = 0.3

    # animate_price_to_expiry("c", S, K, T, r, d, v)
    call_greeks = black_scholes("c", S, K, T, r, d, v)
    put_greeks  = black_scholes("p", S, K, T, r, d, v)

    print(call_greeks)
    print(put_greeks)

    # bugs:
    # ATM call and put delta are not exactly 0.5
    # ATM call and put rho are not same absolute magnitude

    # need d_del_v, charm, weezu