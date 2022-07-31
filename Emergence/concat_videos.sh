# make a file inputs.txt with the filenames like this:
# file inp1.mp4
# file inp2.mp4
# etc.

ffmpeg -f concat -i inputs.txt -vcodec copy -acodec copy Mux1.mp4
