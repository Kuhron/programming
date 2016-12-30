import random

from datetime import datetime
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.stations_by_id = {}
        self.dts = []
        self.day_type_descriptions = {"W":"Weekday", "A":"Saturday", "U":"Sunday/Holiday"}

    def process_entry(self, raw_line):
        line = raw_line.strip().split(",")
        station_id, station_name, date, day_type, ride_count = line
        dt = datetime.strptime(date, "%m/%d/%Y")
        if dt not in self.dts:
            self.dts.append(dt)
        ride_count = int(ride_count)
        if station_id not in self.stations_by_id:
            self.stations_by_id[station_id] = Station(station_id, station_name)
        self.stations_by_id[station_id].add_day_data(dt, day_type, ride_count)


class Station:
    def __init__(self, station_id, station_name):
        self.station_id = station_id
        self.station_name = station_name
        self.day_types = {}
        self.ride_counts = {}
    
    def add_day_data(self, dt, day_type, ride_count):
        self.day_types[dt] = day_type
        self.ride_counts[dt] = ride_count

    def get_ride_count_plot_series(self, max_days, day_type=None):
        xs = sorted([i for i in list(self.ride_counts.keys()) if self.day_types[i] == day_type or day_type in ["N", None]])
        ys = [self.ride_counts[i] for i in xs]
        return xs[:max_days], ys[:max_days], self.station_name

    def get_busiest_date(self):
        dt = max(self.ride_counts, key=lambda x: self.ride_counts[x])
        return dt.strftime("%Y-%m-%d")


def plot_ride_counts(ride_counts_lst, names_lst):
    xs_lst = [i[0] for i in ride_counts_lst]
    ys_lst = [i[1] for i in ride_counts_lst]

    fig, ax = plt.subplots()
    for i in range(len(ys_lst)):
        ys = ys_lst[i]
        name = names_lst[i]
        plt.plot(xs_lst[0], ys, label=name)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=15, fontsize=10)
    plt.legend(loc=2)
    plt.show()
    plt.close()


f = open("Data/CTA_Ridership_L_Station_Entries_Daily_Totals.csv", "r")
lines = f.readlines()
f.close()

data = Data()

header = lines[0].strip().split(",")

max_days = int(input("max days to plot: "))
# chosen_station = random.choice(list(data.stations_by_id.values()))
chosen_station_ids = input("station ids to analyze (separated by spaces): ").split()
day_type = input("day_type (W=Weekday, A=Saturday, U=Sunday/Holiday), N=None(all): ")

for line in lines[1:]:
    if len(data.dts) > max_days:
        break
    if line.split(",")[0] not in chosen_station_ids:
        continue
    data.process_entry(line)

ride_counts_lst = []
names_lst = []
for chosen_station_id in chosen_station_ids:
    chosen_station = data.stations_by_id[chosen_station_id]
    names_lst.append(chosen_station.station_name)
    ride_counts_lst.append(chosen_station.get_ride_count_plot_series(max_days, day_type))
    print("Station {0}, busiest day {1}".format(chosen_station.station_name, chosen_station.get_busiest_date()))

plot_ride_counts(ride_counts_lst, names_lst)