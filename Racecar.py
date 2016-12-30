import random

import numpy as np
import matplotlib.pyplot as plt


class Racecar:
    def __init__(self):
        self.speed = random.uniform(0.01, 0.99)
        self.sd = random.uniform(0.01, 0.1)
        self.position = 0
        self.laps = 0
        self.distance = 0

    def advance(self, track_length):
        self.speed = max(0.01, min(0.99, self.speed + random.normalvariate(0, self.sd)))
        self.position += self.speed
        self.distance += self.speed
        if self.position >= track_length:
            self.laps += 1
            self.position %= track_length


class Station:
    def __init__(self, position):
        self.position = position
        self.history = []

    def measure(self, cars):
        if any(int(car.position) == int(self.position) for car in cars):
            self.history.append(1)
        else:
            self.history.append(0)


class Racetrack:
    def __init__(self, length, n_cars):
        self.length = length
        self.cars = [Racecar() for i in range(n_cars)]
        self.stations = [Station(x) for x in range(self.length)]
        self.time = 0

    def print_car_params(self):
        for i, car in enumerate(self.cars):
            print("car {0:-3d}: initial speed {1:.4f}, sd {2:.4f}".format(i, car.speed, car.sd))

    def advance_cars(self):
        for car in self.cars:
            car.advance(self.length)

    def measure_cars(self):
        for station in self.stations:
            station.measure(self.cars)

    def advance(self, n_periods):
        for i in range(n_periods):
            self.advance_cars()
            self.measure_cars()
            self.time += 1

    def plot_station_histories(self, n_stations):
        for station in random.sample(self.stations, n_stations):
            print(station.position)
            for t in range(len(station.history)):
                if station.history[t] == 1:
                    plt.axvline(t, color="b")
            plt.show()

    def plot_time(self):
        for station in self.stations:
            observations = [t for t, ob in enumerate(station.history) if ob == 1]
            xs = observations
            ys = [station.position] * len(observations)
            plt.scatter(xs, ys, c="k", s=1)
            # plt.plot([station.position] * len(observations), observations, "ok")
        plt.xlim((0, self.time))
        plt.ylim((0, self.length))
        plt.show()

    def plot_distances(self):
        xs = [car.sd for car in self.cars]
        ys = [car.distance for car in self.cars]
        plt.scatter(xs, ys)
        plt.show()


track = Racetrack(100, n_cars=10)
track.print_car_params()
track.advance(1000)
track.plot_time()


