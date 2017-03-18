import math
import random
import html as html_lib
import requests

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


def get_populous_us_city():
    wikipedia_url = "https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population"
    html = requests.get(wikipedia_url).text
    soup = BeautifulSoup(html, "html5lib")
    tables = soup.find_all("table", attrs={"class": "wikitable sortable"})
    table = tables[0]
    trs = table.find_all("tr")
    cities = []
    for tr in trs[1:]:
        tds = tr.find_all("td")
        city_td = tds[1]
        city = city_td.find("a").text
        state_td = tds[2]
        state = state_td.find("a").text
        new_str = u"{}, {}".format(city, state)
        # print(new_str.encode("utf-8"))
        cities.append(new_str)

    assert all(x in cities for x in ["New York, New York", "Chicago, Illinois", "Memphis, Tennessee"])
    return random.choice(cities)


if __name__ == "__main__":
    mode = input("Select mode:\n"
        "1. World\n"
        "2. Continental US (approx.)\n"
        "3. US city over 100,000 people\n")
    if mode == "1":
        print(get_random_latitude_and_longitude(degrees=True))
    elif mode == "2":
        print(get_random_latitude_and_longitude(degrees=True,lat_range=(24.5,49.5),lon_range=(-125,-66)))
    elif mode == "3":
        print(get_populous_us_city())