import csv

METADATA_FP = "MapImageMetadata.csv"
WORLD_METADATA_FP = "WorldMetadata.csv"


def get_image_metadata_dict():
    key_column = "image_name"
    d = {}
    with open(METADATA_FP) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[key_column]
            d[key] = row
    return d


def get_world_metadata_dict():
    key_column = "world_name"
    d = {}
    with open(WORLD_METADATA_FP) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[key_column]
            d[key] = row
    return d


def get_latlon_dict():
    metadata = get_image_metadata_dict()
    d = {}
    for image_name, row in metadata.items():
        latlon00 = (float(row["lat00"]), float(row["lon00"]))
        latlon01 = (float(row["lat01"]), float(row["lon01"]))
        latlon10 = (float(row["lat10"]), float(row["lon10"]))
        latlon11 = (float(row["lat11"]), float(row["lon11"]))
        latlons = [latlon00, latlon01, latlon10, latlon11]
        d[image_name] = latlons
    return d


def get_icosa_distance_tolerance_normalized(image_name):
    # scaling the distance as though planet radius is 1
    metadata = get_image_metadata_dict()[image_name]
    icosa_point_tolerance_km = float(metadata["icosa_point_tolerance_km"])
    world_name = metadata["world_name"]
    planet_radius_km = float(get_world_metadata_dict()[world_name]["planet_radius_km"])
    icosa_distance_tolerance_normalized = icosa_point_tolerance_km / planet_radius_km
    return icosa_distance_tolerance_normalized


if __name__ == "__main__":
    d = get_image_metadata_dict()
    print(d)
