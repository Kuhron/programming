import pickle
import requests
import time
from bs4 import BeautifulSoup


def get_city_url(city_name, state_abbr):
    # e.g., http://www.areavibes.com/savannah-ga/cost-of-living/
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
    all_col_data = {}

    with open("Liveability/cities.txt") as f:
        lines = f.readlines()
    cities = [tuple(x.strip().split(",")) for x in lines]
    
    for city, state in cities:
        key = city + "-" + state
        path = "Liveability/AreaVibeData/{}_col.pickle".format(key)

        try:
            with open(path, "rb") as f:
                col_data = pickle.load(f)
        except FileNotFoundError:
            print("scraping: {}, {}".format(city, state))
            url = get_city_url(city, state)
            col_url = url + "cost-of-living/"
            html = requests.get(col_url).text
            col_data = find_col_tables(html)

            if col_data is None:
                print("unable to find url for {}".format(key))
                continue

            with open(path, "wb") as f:
                pickle.dump(col_data, f)

            time.sleep(1)

        all_col_data[(city, state)] = col_data

    return all_col_data


if __name__ == "__main__":
    all_col_data = get_all_col_data()        

    # sort_key = "Pizza"
    sort_key = "Home Price"

    print("sorted by cost of {} (descending)".format(sort_key))
    for city in sorted(cities, key=lambda x: all_col_data[x][sort_key]["local"], reverse=True):
        print(city, all_col_data[city][sort_key]["local"])