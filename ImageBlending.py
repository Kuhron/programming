import sys

#from PIL import Image as image # install PIL with "python -m pip install pillow"
from PIL import Image

# class Image:
#   def __init__(self,filepath):
#       self.im = image.open(filepath)
#       self.pix = self.im.load()
#       self.x, self.y = self.im.size
#       self.pixels = [[self.pix[c,r] for c in range(self.x)] for r in range(self.y)]

# def blend(Image1, Image2):
#   x = max(Image1.x,Image2.x)
#   y = max(Image1.y,Image2.y)
#   im = image.new("png",(x,y))

def blend(filename1,filename2,alpha_percent):
    s1 = filename1.split(".")
    s2 = filename2.split(".")

    ext = s1[-1]
    if s2[-1] != ext:
        raise TypeError("Both images must be same file extension.")

    f1 = ".".join(s1[:-1])
    f2 = ".".join(s2[:-1])

    im_a = Image.open(DIRECTORY+"\\"+filename1)
    im_b = Image.open(DIRECTORY+"\\"+filename2)
    
    im_ab = Image.blend(im_a,im_b,alpha=float(alpha_percent)/100.0)
    full_filepath = DIRECTORY+"\\"+f1+"_"+f2+"_alpha"+str(alpha_percent)+"."+ext
    print("Saved to %s" % full_filepath)
    im_ab.save(full_filepath)

DIRECTORY = "C:\\Users\\Wesley\\Pictures\\Databending"

args = sys.argv
if len(args) != 4:
    print("usage: python \"Image Blending.py\" [filepath 1] [filepath 2] [alpha \\in [0,100]]")
    sys.exit()

#blend("Blend_1a.png","Blend_1b.png",50)
blend(*args[1:])