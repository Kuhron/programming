import argparse
import math
import msvcrt
import os
import random
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


def clear_file(filepath):
    f = open(filepath, "w")
    f.write("")
    f.close()

def update_file(filepath, t_0):
    f = open(filepath, "r")
    lines = f.readlines() # slow for large file?
    if len(lines) > 0:
        last_line = get_last_line(lines)
        val = float(last_line.split(",")[-1])
    else:
        val = 0
    f.close()

    f = open(filepath, "a")
    t = time.time() - t_0
    # new_val = val + random.random() - 0.5
    new_val = val + random.normalvariate(0,1)
    # new_val = val + random.paretovariate(5) * random.choice([-1,1])
    f.write("{0:.4f},{1:.4f}\n".format(t, new_val))
    f.close()

    return new_val

def get_last_line(lines):
    if len(lines) == 0:
        return None
    last_line_index = -1
    last_line = lines[last_line_index]
    while last_line == "\n" or last_line == "":
        last_line_index -= 1
        last_line = lines[last_line_index]
    return last_line

def animate(i):
    lines = open('randomtradingdata.txt','r').readlines()
    x_range = 10 # seconds
    try:
        last_line = get_last_line(lines)
        last_x = float(last_line.split(",")[0]) if last_line else 0
    except ValueError:
        last_x = 0
    xs = []
    ys = []
    for line in lines:
        #print("line:",line)
        try:
            if line != "\n" and line != "":
                x, y = line.split(',')
                if len(line) > 1 and float(x) >= last_x - x_range:
                    xs.append(x)
                    ys.append(y)
        except ValueError:
            print("line raising error:",line)
            sys.exit()
    ax1.clear()
    ax1.plot(xs, ys)

def trade(val, trades, strategy):
    if strategy == "manual":
        if msvcrt.kbhit():
            key = ord(msvcrt.getch())
            if key in [72,80]: # up and down arrows
                if key == 72:
                    trades.buy(val)
                    # amt = input("Amount to buy: ")
                    # try:
                    #     for i in range(int(amt)):
                    #         trades.buy(val)
                    # except:
                    #     print("invalid input")
                elif key == 80:
                    trades.sell(val)
                    # amt = input("Amount to sell: ")
                    # try:
                    #     for i in range(-1*int(amt)):
                    #         trades.sell(val)
                    # except:
                    #     print("invalid input")
            elif key == 13:
                print("user terminated, raising KeyboardInterrupt")
                raise KeyboardInterrupt
    else:
        dec = strategy.receive_decision(val, trades.position)
        if dec == "buy":
            trades.buy(val,print_status=False)
        elif dec == "sell":
            trades.sell(val,print_status=False)
        elif dec == "end":
            raise KeyboardInterrupt
    trades.update(val)
    return trades


class Strategy:
    def __init__(self):
        self.t_0 = time.time()
        self.time_limit = 600
        self.offset = 25
        self.midpoint = 0
        self.max_operating_position = 2000
        self.max_position = 10000
        self.operating_position = 0
        self.operating_position_offset = 0

    def receive_decision(self, val, position):
        self.operating_position = position + self.operating_position_offset
        while abs(self.operating_position) >= self.max_operating_position:
            self.operating_position_offset -= self.operating_position
            self.operating_position = position + self.operating_position_offset

        if time.time() >= self.t_0 + self.time_limit:
            return "end"

        if self.operating_position == 0:
            self.midpoint = val
            return random.choice(["buy","sell"])
        # elif abs(self.operating_position) >= self.max_operating_position:
        #     return "none"
        else:
            if val < self.midpoint - self.offset and position < self.max_position:
                return "buy"
            elif val > self.midpoint + self.offset and position > -1*self.max_position:
                return "sell"
            else:
                return "none"


class Trades:
    def __init__(self, string_from_file=None):
        self.tradelist = []
        self.permanent_tradelist = []
        self.position = 0
        self.position_list = []
        self.permanent_pnl = 0
        self.pnl_list = []
        if string_from_file:
            self.parse_string_from_file(string_from_file)

    def buy(self, val, print_status=True):
        self.tradelist.append((1, val))
        self.permanent_tradelist.append((1, val))
        self.position += 1
        self.position_list.append(self.position)
        self.pnl_list.append(self.get_pnl(val))
        if print_status:
            print("bought at {0:.4f}".format(val))
            print("{2:3.4f}, Position {0}, PnL {1:.2f}.".format(self.position, self.pnl_list[-1], val))

    def sell(self, val, print_status=True):
        self.tradelist.append((-1, val))
        self.permanent_tradelist.append((-1, val))
        self.position -= 1
        self.position_list.append(self.position)
        self.pnl_list.append(self.get_pnl(val))
        if print_status:
            print("sold at   {0:.4f}".format(val))
            print("{2:3.4f}, Position {0}, PnL {1:.2f}.".format(self.position, self.pnl_list[-1], val))

    def get_pnl(self, val):
        trade_pnls = [trade[0] * (val - trade[1]) for trade in self.tradelist]
        pnl = sum(trade_pnls) + self.permanent_pnl
        return pnl

    def update(self,val,end=False):
        if (self.position == 0 or end) and self.tradelist != []:
            self.permanent_pnl = self.get_pnl(val)
            f = open("randomtradingtradelist.txt","a")
            f.write("\n".join([repr(i) for i in self.tradelist])+"\n")
            print("tradelist written to file")
            f.close()
            self.tradelist = []

    def end(self):
        if self.permanent_tradelist != []:
            self.update(self.permanent_tradelist[-1][-1],end=True)
        else:
            return

    def parse_string_from_file(self, s):
        lines = s.split("\n")
        for line in lines:
            if line == "" or line == "\n":
                pass
            tuple_contents = line[1:-1]
            try:
                action,val = tuple_contents.split(",")
                if action == "1":
                    self.buy(float(val),print_status=False)
                elif action == "-1":
                    self.sell(float(val),print_status=False)
            except ValueError:
                continue

def print_trade_analysis():
    f = open("randomtradingtradelist.txt","r")
    final_trades = Trades(f.read())
    f.close()

    plt.plot(final_trades.position_list)
    plt.title("position")
    plt.show()
    plt.close()

    plt.plot(final_trades.pnl_list)
    plt.title("pnl")
    plt.show()
    plt.close()

    trade_prices = [i[-1] for i in final_trades.permanent_tradelist]
    trade_directions = [i[0] for i in final_trades.permanent_tradelist]
    buy_sell_colors = [("b" if i == 1 else "r") for i in trade_directions]
    plt.scatter(range(len(final_trades.permanent_tradelist)), trade_prices, c=buy_sell_colors, lw=0)
    plt.plot(range(len(final_trades.permanent_tradelist)), trade_prices, "k")
    plt.title("trades")
    plt.show()
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("--generator", action="store_true")
args = parser.parse_args()

data_file = "randomtradingdata.txt"

if args.generator:
    t_0 = time.time()
    trades = Trades()
    strategy_choice = input("How do you want to trade?\n \
        1. manually\n \
        2. by collecting a spread around a midpoint\n")
    if strategy_choice == "1":
        print("up arrow to buy, down arrow to sell, enter to stop trading")
        strategy = "manual"
    elif strategy_choice == "2":
        strategy = Strategy()
    while True:
        try:
            val = update_file(data_file, t_0)
            trades = trade(val, trades, strategy)
            time.sleep(0.005)
        except KeyboardInterrupt:
            trades.end()
            print("generator process terminating. close the plot to cleanly exit the plotter process")
            break
else:
    try:
        clear_file(data_file)
        clear_file("randomtradingtradelist.txt")
        input("files cleared, press enter to continue")
        plt.close()

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)

        process = subprocess.Popen(["python","RandomTrading.py","--generator"])
        ani = animation.FuncAnimation(fig, animate, interval=100)
        plt.show()

        raise KeyboardInterrupt

        plt.close()

    except KeyboardInterrupt:
        print("plotter process terminating")
        process.wait()

    print("analyzing trading")
    print_trade_analysis()



























