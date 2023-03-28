raise Exception("Watch out for high memory usage! Use ffmpeg instead (see FolderToGif.sh).")

# https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
from PIL import Image
import os


def make_gif(frame_folder):
    frames = [Image.open(os.path.join(frame_folder, fp)) for fp in sorted(os.listdir(frame_folder))]
    frame_one = frames[0]
    output_fp = "output.gif"
    total_length = 6 * 1000
    frame_duration = int(round(total_length / len(frames)))
    frame_one.save(output_fp, format="GIF", append_images=frames[1:], save_all=True, duration=frame_duration, loop=0)
    print(f"saved to {output_fp}")


if __name__ == "__main__":
    make_gif("Images/HueRotation/20220330_230210/")

