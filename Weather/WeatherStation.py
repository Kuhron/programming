from datetime import datetime
import json
import os
import pickle
import random
import requests
import time

import matplotlib.pyplot as plt


API_KEY = "7e2cadfd5408d1699dda399172bb8442"  # from my account on the website, good for 600 calls in any 10 minutes (1 Hz) (free account)
COORDS_LIST = [(lat, lon) for lat in range(-50, 51, 5) for lon in range(-180, 176, 45)]
DATA_DIR = "C:/Users/Wesley/Desktop/Programming/Weather/OpenWeatherMap_data/"


def save_object(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x


def update_data(coords):
    print("updating data for coords", coords)
    lat, lon = get_lat_lon_from_coords(coords)

    lat_lon_str = get_lat_lon_str(lat, lon)

    req = requests.get("http://api.openweathermap.org/data/2.5/weather?lat={0}&lon={1}&APPID={2}".format(lat, lon, API_KEY))
    now_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = DATA_DIR + "{now_str}_{lat_lon_str}.pickle".format(**locals())
    save_object(req, path)

    print("updated  data for coords:", coords, "\n", req.content)
    print("")

    return req


def get_lat_lon_from_coords(coords):
    lat, lon = coords
    try:
        lat = float(lat)
        lon = float(lon)
    except:
        print("error converting coords to float:", coords)
        raise
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180

    return lat, lon


def get_lat_lon_str(lat, lon):
    lat_direction = "N" if lat >= 0 else "S"
    lon_direction = "E" if lon >= 0 else "W"
    return str(abs(lat)) + lat_direction + "_" + str(abs(lon)) + lon_direction


def matches_coords(path, lat, lon):
    lat_lon_str = get_lat_lon_str(lat, lon)
    return lat_lon_str in path


class WeatherData:
    def __init__(self, loaded_object):
        self.loaded_object = loaded_object
        self.__dict__.update(json.loads(loaded_object.content.decode("utf-8")))


def plot_data_attribute(attr, coords_list, by_time=True):
    for coords in coords_list:
        lat, lon = get_lat_lon_from_coords(coords)
        objs = []
        for dirpath, dirnames, filenames in os.walk(DATA_DIR):
            for path in filenames:
                if matches_coords(path, lat, lon):
                    obj = WeatherData(load_object(DATA_DIR + path))
                    objs.append(obj)
        s_objs = sorted(filter(lambda x: hasattr(x, "dt"), objs), key=lambda x: x.dt)
        dts = [obj.dt for obj in s_objs]
        series = [obj.main.get(attr) for obj in s_objs]
        print("series to plot:", series)
        lab = "{0}, {2} (id {1})".format(s_objs[0].name, s_objs[0].id, s_objs[0].sys["country"])
        if by_time:
            plt.plot(dts, series, label=lab)
        else:
            plt.plot(range(len(series)), series, label=lab)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1))
    plt.show()


if __name__ == "__main__":
    coords_to_plot = random.sample(COORDS_LIST, 3)
    plot_data_attribute("temp", coords_to_plot, by_time=False)
    plot_data_attribute("pressure", coords_to_plot, by_time=False)



