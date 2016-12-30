# copied from http://stackoverflow.com/questions/5419888/reading-from-a-frequently-updated-file
# refactored slightly

import time
        

def follow(filepath):
    with open(filepath) as f:
        # f.seek(0,2)  # goes to end of file
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
            else:
                yield line


if __name__ == '__main__':
    filepath = "a.txt"
    loglines = follow(filepath)
    for line in loglines:
        print(line)