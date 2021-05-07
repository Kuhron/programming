import math
import random
import html as html_lib
import requests
import webbrowser
import sys

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


inf = float("inf")
real_range = (-1*inf,inf)


def in_range(range_tuple,value):
    if len(range_tuple) != 2 or range_tuple[0] > range_tuple[1]:
        raise ValueError("Invalid range tuple")
    return value >= range_tuple[0] and value <= range_tuple[1]


def get_random_latitude_and_longitude(degrees=False,lat_range=real_range,lon_range=real_range):
    lat,lon = None,None
    while lat is None or lon is None or (lat_range is not None and not in_range(lat_range,lat)) \
        or (lon_range is not None and not in_range(lon_range,lon)):
        lon = random.uniform(-1,1)*(180 if degrees else math.pi)
        cos_theta = random.uniform(-1,1)
        theta = (math.acos(cos_theta) - math.pi * 1.0/2)
        lat = theta * (90/(math.pi*1.0/2) if degrees else 1)

    return (lat,lon)


def get_populous_us_cities(n_cities):
    # current page, subject to breaking randomly
    # wikipedia_url = "https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population"

    # static revision, if I can get it to work for one of them
    wikipedia_url = "https://en.wikipedia.org/w/index.php?title=List_of_United_States_cities_by_population&oldid=854477638"
    print("using revision as of 2018-08-11")

    html = requests.get(wikipedia_url).text
    soup = BeautifulSoup(html, "html5lib")
    tables = soup.find_all("table", attrs={"class": "wikitable sortable"})
    table = tables[0]
    trs = table.find_all("tr")
    cities = []
    for tr in trs[1:]:
        tds = tr.find_all("td")
        city_td = tds[1]
        city = city_td.find("a").text.strip()
        state_td = tds[2]
        state = state_td.text.replace("\u00A0", "").strip()  # remove &nbsp;
        new_str = u"{}, {}".format(city, state)
        # print(new_str.encode("utf-8"))
        cities.append(new_str)

    assert all(x in cities for x in ["New York, New York", "Chicago, Illinois", "Memphis, Tennessee"]), cities
    return random.sample(cities, n_cities)


def get_world_city_located_by_population():
    input_fp = "/home/wesley/GithubGists/world_cities_data/worldcities_clean.csv"
    df = pd.read_csv(input_fp)
    columns_i_care_about = ["city", "country", "population"]
    df = df[columns_i_care_about]
    population_probability_vector = df["population"] / df["population"].sum()
    choice_index = np.random.choice(df.index, p=population_probability_vector)
    row = df.loc[choice_index, :]
    s = f"{row['city']}, {row['country']}"
    return s


def get_us_location_weighted_by_population():
    # level = "state"
    # level = "county"
    level = "city"
    pop_data = get_us_location_population_data()
    d = pop_data[level]
    names, pops = zip(*d.items())
    p = np.array(pops) / sum(pops)
    chosen_index = np.random.choice(np.arange(len(names)), p=p)
    chosen = names[chosen_index]
    return chosen


def get_us_location_population_data():
    data_fp = "Census_2010-2019_all.csv"
    df = pd.read_csv(data_fp, engine='python')  # without engine="python", get UnicodeDecodeError
    name_colname = "NAME"
    state_colname = "STNAME"
    pop_colname = "POPESTIMATE2019"
    names = df[name_colname]
    states = df[state_colname]
    pops = df[pop_colname]

    is_state = lambda name, state: name == state
    state_pops = {}
    new_names = []
    for name, state, pop in zip(names, states, pops):
        if is_state(name, state):
            state_pops[name] = pop
        else:
            new_names.append(name)
    names = new_names

    is_county = lambda name: name.endswith("County") and not name.startswith("Balance of")
    county_pops = {}
    new_names = []
    for name, state, pop in zip(names, states, pops):
        if is_county(name):
            tup = (name, state)
            county_pops[tup] = pop
        else:
            new_names.append(name)
    names = new_names

    city_pops = {}  # include county residues as if they were cities ("Balance of X County")
    for name, state, pop in zip(names, states, pops):
        assert not is_state(name, state)
        assert not is_county(name)
        name = name.replace("Balance of ", "Unincorporated places in ")
        tup = (name, state)
        city_pops[tup] = pop

    return {"state": state_pops, "county": county_pops, "city": city_pops}


def open_location_in_google_maps(lat, lon):
    zoom_level = 6  # int, bigger is zoomed farther in
    url = "http://www.google.com/maps/place/{lat},{lon}/@{lat},{lon},{zoom_level}z".format(**locals())
    webbrowser.open(url)


def confirm(string):
    x = input(string + " (y/n, default = n)")
    return x.strip().lower() == "y"


if __name__ == "__main__":
    args = sys.argv

    try:
        open_in_browser = args[2] == "y"
    except IndexError:
        open_in_browser = None
    confirm_open = lambda: open_in_browser if open_in_browser is not None else confirm("open in browser?")

    try:
        mode = args[1]
    except IndexError:
        mode = input("Select mode:\n"
            "1. World\n"
            "2. Continental US (approx.)\n"
            "3. US cities over 100,000 people\n"
            "4. US location weighted by population\n"
            "5. World city weighted by population\n")
    if mode == "1":
        loc = get_random_latitude_and_longitude(degrees=True)
        print(loc)
        if confirm_open():
            open_location_in_google_maps(*loc)
    elif mode == "2":
        loc = get_random_latitude_and_longitude(degrees=True,lat_range=(24.5,49.5),lon_range=(-125,-66))
        print(loc)
        if confirm_open():
            open_location_in_google_maps(*loc)
    elif mode == "3":
        n_cities = int(input("How many cities? "))
        for city in get_populous_us_cities(n_cities):
            print(city)
    elif mode == "4":
        loc = get_us_location_weighted_by_population()
        print(loc)
    elif mode == "5":
        loc = get_world_city_located_by_population()
        print(loc)

# TODO: rang road trip, between cities and/or points


