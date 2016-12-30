import time

from WeatherStation import COORDS_LIST, update_data


def run_collector(coords_list):
    print("running weather data collector")
    # wait at least 2 seconds between calls no matter what, and only update weather at each location once every 10 minutes
    wait_per_loc = max(2.0, 600.0 / len(coords_list)) + 0.1
    print("waiting {0} seconds for each of {1} locations".format(wait_per_loc, len(coords_list)))
    while True:
        for coords in coords_list:
            time.sleep(wait_per_loc)
            update_data(coords)
        print("waiting 10 seconds until next cycle\n.\n..\n...")



run_collector(COORDS_LIST)