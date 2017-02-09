import html as html_lib
import math
import pickle
import re
import requests
import time

from bs4 import BeautifulSoup


def get_city_url(city_name, state_abbr):
    s = city_name.lower().replace(" ", "+") + "-" + state_abbr.lower()
    return "http://www.areavibes.com/" + s + "/"


def find_col_tables(html):
    soup = BeautifulSoup(html, "html5lib")

    tables = soup.find_all("table", attrs={"class": "av-default col-cmp"})
    if len(tables) != 6:
        return None

    d = {}

    for table in tables:
        trs = table.find_all("tr")
        for tr in trs[1:]:
            tds = tr.find_all("td")
            item = tds[0].text
            local = float(tds[1].text.replace("$", "").replace(",", ""))
            national = float(tds[2].text.replace("$", "").replace(",", ""))
            diff = 0.01 * float(tds[3].text.replace("%", "").replace(",", ""))
            d[item] = {"local": local, "national": national, "diff": diff}

    return d


def get_all_col_data():
    return get_all_data_generic("col", "cost-of-living", find_col_tables)


def get_all_data_generic(designation_str, url_extension, find_tables_func):
    all_col_data = {}

    with open("Liveability/cities.txt") as f:
        lines = f.readlines()
    cities = [tuple(x.strip().split(",")) for x in lines]
    
    for city, state in cities:
        key = city + "-" + state
        path = "Liveability/AreaVibeData/{}_{}.pickle".format(key, designation_str)

        try:
            with open(path, "rb") as f:
                col_data = pickle.load(f)
        except FileNotFoundError:
            print("scraping: {}, {}".format(city, state))
            url = get_city_url(city, state)
            col_url = url + url_extension + "/"
            html = requests.get(col_url).text
            col_data = find_tables_func(html)

            if col_data is None:
                print("unable to find data for {} (url {})".format(key, col_url))
                continue

            with open(path, "wb") as f:
                pickle.dump(col_data, f)

            time.sleep(1)

        all_col_data[(city, state)] = col_data

    return all_col_data


def find_weather_tables(html):
    soup = BeautifulSoup(html, "html5lib")

    tables = soup.find_all("table", attrs={"class": re.compile("av-default")})
    if len(tables) != 6:
        # there are two extra tables that have class "av-default has-own-city-last" but I can't get BeautifulSoup not to match them
        print(len(tables))
        return None

    d = {}

    table = tables[0]
    trs = table.find_all("tr")
    for tr in trs[1:]:
        tds = tr.find_all("td")
        month = tds[0].text
        min_temp = float(tds[1].text.split(html_lib.unescape("&deg;"))[0].replace("n/a", "nan"))
        max_temp = float(tds[2].text.split(html_lib.unescape("&deg;"))[0].replace("n/a", "nan"))
        avg_temp = float(tds[3].text.split(html_lib.unescape("&deg;"))[0].replace("n/a", "nan")) if not (
            math.isnan(min_temp) or math.isnan(max_temp)) else float("nan")
        precipitation = float(tds[4].text.split("\"")[0].replace("n/a", "nan"))
        d[month] = {"min_temp_degf": min_temp, "max_temp_degf": max_temp, "avg_temp_degf": avg_temp, "precipitation_in": precipitation}

    table = tables[1]
    trs = table.find_all("tr")
    for tr in trs[1:]:
        tds = tr.find_all("td")
        item = tds[0].text
        local = float(tds[1].text.replace(",", ""))
        state = float(tds[2].text.replace(",", ""))
        national = float(tds[3].text.replace(",", ""))
        d[item] = {"local": local, "state": state, "national": national}

    for table in tables[2:4]:
        trs = table.find_all("tr")
        for tr in trs[1:]:
            tds = tr.find_all("td")
            item = tds[0].text
            value_text = tds[1].text
            value = float("nan") if value_text == "" else float(value_text) if "%" not in value_text else (0.01 * float(value_text.replace("%", "")))
            item = "air_quality_days_measured" if item == "Days measured" else item
            d[item] = value

    return d


def get_all_weather_data():
    return get_all_data_generic("weather", "weather", find_weather_tables)


def get_all_data():
    d = {}

    data_dicts = [
        get_all_col_data(),
        get_all_weather_data(),
    ]

    for sub_d in data_dicts:
        for k, v in sub_d.items():
            if k not in d:
                d[k] = {}
            d[k].update(v)

    return d


if __name__ == "__main__":
    all_data = get_all_data()

    # get_sort_value = lambda city: all_data[city]["January"]["avg_temp_degf"]
    # get_sort_value = lambda city: all_data[city]["July"]["avg_temp_degf"]
    # get_sort_value = lambda city: all_data[city]["December"]["precipitation_in"]

    cities = [k for k in all_data.keys() if not math.isnan(get_sort_value(k))]
    for city in sorted(cities, key=get_sort_value, reverse=True):
        print(city, get_sort_value(city))