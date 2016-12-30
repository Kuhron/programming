import os
import urllib
import requests

url = urllib.request.URLopener()
# url.retrieve("http://randomsite.com/file.gz", "file.gz")  # example

destination_dir = r"C:\Users\Wesley\Desktop\Programming\MorseCode"
audio_dir = destination_dir + "\\" + r"Audio"
text_dir = destination_dir + "\\" + r"Text"

url_base = "http://www.arrl.org"

def get_mp3s_from_html(text):
    return get_by_delim(text, "\"><u>MP3")

def get_texts_from_html(text):
    return get_by_delim(text, "\"><u>Text")

def get_by_delim(text, delim):
    return [x.split("\"")[-1] for x in text.split(delim)[:-1]]

# for wpm in [5, 7.5, 10, 13, 15, 18, 20, 25, 30, 35, 40]:
for wpm in [15, 25, 40]:
    page = requests.get("http://www.arrl.org/{}-wpm-code-archive".format(wpm))
    mp3s = get_mp3s_from_html(page.text)
    texts = get_texts_from_html(page.text)

    interleaved = [path for pair in zip(mp3s, texts) for path in pair]
    is_mp3 = True
    for x in interleaved:
        destination = (audio_dir + "\\" + x.split("/")[-1]) if is_mp3 else (text_dir + "\\" + x.split("/")[-1])
        if not os.path.isfile(destination):
            url.retrieve(x, destination)
        is_mp3 = not is_mp3
