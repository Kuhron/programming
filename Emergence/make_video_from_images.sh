# thanks to
# - https://askubuntu.com/questions/610903/how-can-i-create-a-video-file-from-a-set-of-jpg-images
# - http://ffmpeg.org/ffmpeg-all.html#image2-1

ffmpeg -framerate 25 -i 2022-07-31-050556-i%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
