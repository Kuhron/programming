from hashlib import sha3_512


def get_hash_of_file(fp):
    with open(fp, "rb") as f:
        contents = f.read()
    return sha3_512(contents).hexdigest()


def files_are_same(fp1, fp2):
    return get_hash_of_file(fp1) == get_hash_of_file(fp2)


# TODO figure out how to move/rename files when they conflict in locations, ask user for input gratuitously rather than doing stuff that they may not want

