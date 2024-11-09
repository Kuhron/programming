import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from csv import DictReader

hexes = {
    1: "#FFFFE0",
    2: "#FF1493",
    3: "#E6E6FA",
    4: "#FA8072",
    5: "#8A2BE2",
    6: "#006400",
    7: "#FAF0E6",
    8: "#48D1CC",
    9: "#DC143C",
    10: "#FFC0CB",
    11: "#FFEBCD",
    12: "#663399",
    13: "#000080",
    14: "#7FFFD4",
    15: "#ADFF2F",
    16: "#2E8B57",
    17: "#191970",
    18: "#FFFF00",
    19: "#D2691E",
    20: "#B0C4DE",
}

names = {}
hex_to_name_css4 = {y:x for x,y in mcolors.CSS4_COLORS.items()}
for i, h in hexes.items():
    names[i] = hex_to_name_css4[h]

print(names)

fp = "ColorCardCategorizationData.csv"
with open(fp) as f:
    reader = DictReader(f)
    rows = [x for x in reader]
print(rows)

g = nx.Graph()
for i, h in hexes.items():
    g.add_node(i)
    g.nodes[i]["color"] = h
for 

nx.draw(g, node_color=[node["color"] for label, node in g.nodes.items()], edgecolors="k")
plt.show()
