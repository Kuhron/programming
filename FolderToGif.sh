# to avoid using PIL because of memory hogging

# imagemagick also uses too much memory (looks like it's loading all the images into memory first, which is the same problem PIL has)
# convert -delay 20 -loop 0 *.jpg myimage.gif
# https://askubuntu.com/questions/648244/how-do-i-create-an-animated-gif-from-still-images-preferably-with-the-command-l
# ffmpeg answer: https://askubuntu.com/a/1102183/638487
ffmpeg -framerate 60 -pattern_type glob -i "*.jpg" -vf scale=1024:-1 out.gif

