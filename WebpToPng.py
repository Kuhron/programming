from PIL import Image
import sys
import os

_, inp_fp = sys.argv
print(f"converting {inp_fp} to .png")
assert inp_fp.endswith(".webp"), inp_fp
assert inp_fp.count(".webp") == 1
outp_fp = inp_fp.replace(".webp", ".png")
if os.path.exists(outp_fp):
    raise Exception(f"would overwrite {outp_fp}, exiting")
im = Image.open(inp_fp)
im.save(outp_fp, "png")
print(f"converted to {outp_fp}")
