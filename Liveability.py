import pandas as pd
import numpy as np


def get_column_score(df, col):
    ser = df[col]
    if col == "average_low_temp_january":
        return (ser - 32) * (ser < 32)
    if col == "average_high_temp_july":
        return (90 - ser) * (ser > 90) + (ser - 70) * (ser < 70)
    elif col == "metro_population":
        return -1 * abs(np.log(ser) - np.log(200000))
    elif col == "murders_per_100k_annual":
        return -ser
    else:
        return np.nan


def get_score_df(df):
    score_cols = []
    for col in df.columns.values:
        score_col = col + "_score"
        score_cols.append(score_col)
        df[score_col] = get_column_score(df, col)

    return df[score_cols]


def get_score(score_df):
    assert all("_score" in x for x in score_df.columns.values)
    return score_df.sum(axis=1)


if __name__ == "__main__":
    d = {
        "name": "Anytown",
        "average_low_temp_january": 20,
        "average_high_temp_july": 85,
        "metro_population": 500000,
        "murders_per_100k_annual": 25,
    }
    d_list = [d]
    test_df = pd.DataFrame(d_list)
    test_df = test_df.set_index("name")
    score_df = get_score_df(test_df)
    print(score_df)
    score = get_score(score_df)
    print(score)


    # dump of test code

    import datetime
    import requests
    import traceback
    from pydap.client import open_url

    dt = datetime.datetime(2014, 7, 1, 0, 0, 0)
    ym_str = dt.strftime("%Y%m")
    ymd_str = dt.strftime("%Y%m%d")

    # url = 'http://nomads.ncdc.noaa.gov/dods/NCEP_NARR_DAILY/197901/197901/narr-a_221_197901dd_hh00_000'  # doctype, because this refers to the info page
    url = "https://nomads.ncdc.noaa.gov/dods/NCEP_NARR_DAILY/{0}/{1}/narr-a_221_{1}_0000_000".format(ym_str, ymd_str)  # works! probably earliest dates have problems
    # url = 'http://test.opendap.org/dap/data/nc/coads_climatology.nc'  # sometimes works, sometimes 500 error
    # url = 'http://dapper.pmel.noaa.gov/dapper/argo/argo_all.cdp'  # doctype, because it actually returns an HTML error page
    # url = 'http://nomads.ncdc.noaa.gov/thredds/dodsC/nam218/201111/20111102/nam_218_20111102_0000_000.grb'  # doctype, because returns an error page
    # url = "http://nomads.ncep.noaa.gov:9090/dods/nam/nam20111206/nam1hr_00z"  # error page


    try:
        data = open_url(url)
        # print(type(data))
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print("----")
        page = requests.get(url)
        print(page)
        print(page.text[:1000])

    # print(data.keys())
    tmp2m = data['tmp2m']
    # print(type(tmp2m))
    # print(tmp2m.keys())
    a = tmp2m[:, 0, 0]
    print(a)
    print(dir(a))
