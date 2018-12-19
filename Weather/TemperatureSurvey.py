import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_fp = "Temperature Preferences Responses - Form Responses 1.csv"

df = pd.DataFrame.from_csv(data_fp)

# some cleaning
df = df[df["data_mistake"].isnull()]  # quick and dirty but there are false positives that need to be put back in
# need to switch the temperature scales for the people who made mistakes there
# don't remove the response from Saskatchewan (who still lives there) that has really low preferences, because it is more believable for that one to be Celsius (as reported) than Fahrenheit; thus I would bet that it is not a mistake

all_temps = set(list(df["lowF"]) + list(df["optimumF"]) + list(df["highF"]))

# print(df)
plt.title("low, opt, and high temps in deg F")
minx = min(all_temps)
maxx = max(all_temps)
rangex = maxx - minx
dtemp_in_bin = 2  # deg F
# round min and max bin bounds such that each bin is centered on a multiple of dtemp_in_bin
minbin = int(dtemp_in_bin * (-0.5 + int(minx / dtemp_in_bin)))
maxbin = int(dtemp_in_bin * (1.5 + int(maxx / dtemp_in_bin)))  # hist bins sequence requires both endpoints
bins = range(minbin, maxbin, dtemp_in_bin)
plt.subplot(3, 1, 1)
plt.hist(df["lowF"], color="blue", bins=bins)
plt.xlim(minx, maxx)
plt.subplot(3, 1, 2)
plt.hist(df["optimumF"], color="green", bins=bins)
plt.xlim(minx, maxx)
plt.subplot(3, 1, 3)
plt.hist(df["highF"], color="red", bins=bins)
plt.xlim(minx, maxx)
plt.show()

plt.title("individual responses")


plt.title("range in deg F")
plt.hist(df["rangeF"], bins=100)
plt.show()

plt.title("optimum in interval [0, 1]")
plt.hist(df["opt01"], bins=100)
plt.show()
