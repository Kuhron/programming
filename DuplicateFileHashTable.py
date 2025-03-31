from pathlib import Path
from hashlib import sha256
import argparse
from collections import defaultdict



def dir_path(path_str: str):
    path = Path(path_str).resolve()  # enforce absolute path
    if path.is_dir():
        return path
    raise argparse.ArgumentTypeError(
        f"{path} not valid, should be a directory")


def positive_int(x: int):
    x = int(x)
    if x <= 0:
        raise ValueError(f"int must be positive, got {x}")
    return x


def content_hash(fp):
    with open(fp, "rb") as f:
        return sha256(f.read()).hexdigest()


def two_fps_equal(fp1, fp2):
    with open(fp1, "rb") as f1:
        b1 = f1.read()

    with open(fp2, "rb") as f2:
        b2 = f2.read()

    return b1 == b2


def all_fps_equal(fps):
    with open(fps[0], "rb") as f0:
        b0 = f0.read()

    for fp in fps[1:]:
        with open(fp, "rb") as f:
            b = f.read()
        if b != b0:
            return False
    return True


def get_hash_dict(fps):
    d = defaultdict(list)
    for fp in fps:
        h = content_hash(fp)
        d[h].append(fp)
    return d


def check_no_collisions(d):
    for h, fps in d.items():
        if len(fps) > 1:
            assert all_fps_equal(fps), f"collision! (should be astronomically unlikely) {h = }; {fps = }"


def print_hash_dict(d, min_n=1):
    for h, fps in d.items():
        if len(fps) >= min_n:
            print(h)
            for fp in fps:
                print(fp)
            print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=dir_path,
                        help="the path to the directory where the text's video and audio are stored")
    parser.add_argument("-m", type=positive_int, default=1, help="the minimum number of copies that need to be present for a given hash to be shown")

    args = parser.parse_args()

    dir_path = args.dir_path
    fps = [x for x in dir_path.glob("**/*") if x.is_file()]
    d = get_hash_dict(fps)
    check_no_collisions(d)
    print_hash_dict(d, min_n=args.m)

