from PIL import Image
from PIL import gifmaker
sequence = []
# im = Image.open(....)

frames = [frame.copy() for frame in ImageSequence.Iterator(im)]

fp = open("out.gif", "wb")
gifmaker.makedelta(fp, frames)
fp.close()