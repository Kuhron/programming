import json

with open("ParamConfig.json") as f:
    d = json.load(f)

# generate python code to load params to variables
for k in sorted(d):
    print("{0} = params[\"{0}\"]".format(k))

