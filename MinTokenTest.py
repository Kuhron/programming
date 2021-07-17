import random
import re
import thai_segmenter.sentence_segmenter
import thai_segmenter.word_processing
from thai_segmenter import tokenize


def get_random_sample_text():
    input_fp = "moby-dick.txt"
    with open(input_fp) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    sample_lines = random.sample(lines, 20)
    s = "\n".join(sample_lines)
    s = re.sub("[^\w]","",s).lower()
    return s


def pre_tokenize_text(s, words):
    # returns string where each of the target words has been surrounded by spaces
    # treats longest words first so we don't get things like "whose" being made into "who se" because it contains one of the shorter words
    words_longest_first = sorted(words, key=len, reverse=True)
    for w in words_longest_first:
        s = s.replace(w, " "+w+" ").replace("  ", " ")
    return s


if __name__ == "__main__":
    # text = get_random_sample_text()
    # words_to_identify = ["the", "they", "them", "of", "for", "so", "some", "sometimes", "something", "someone", "somebody", "awesome", "what", "was", "were", "we", "me", "my"]
    # pre_tokenized = pre_tokenize_text(text, words_to_identify)
    # print(pre_tokenized)

    thai_text = "แล้วผู้บ่าวกะตกแลงมากะมีการส่งสาวเข้าหอ เข้าห้องหอซั้นแหล่วเพื่อนเจ้าบ่าวซุมขึ้นไป ขึ้นไปคะเจ้าเล่นสาว ตอนไปลักไก่นิละ ซุมนี้ละไปส่ง"  # example from Min
    thai_text = thai_text.replace(" ", "")
    # custom_dict_fp = "MinTokenTestLexitron.txt"
    # with open(custom_dict_fp) as f:
    #     custom_words = [l.strip() for l in f.readlines()]
    # custom_word_processing = thai_segmenter.word_processing.word_processing(dict_file_paragraph=custom_dict_fp, dict_file_words=None)
    # custom_segmenter = thai_segmenter.sentence_segmenter.sentence_segmenter(custom_dict=custom_words)
    # tokens = tokenize(thai_text, custom_segmenter)
    tokens = tokenize(thai_text)
    text_with_spaces = " ".join(tokens)
    mistake_words = {
        "คะ เจ้า": "คะเจ้า",
        "แหล่ ว": "แหล่ว",
        "ผู้ บ่าว": "ผู้บ่าว",
    }
    print(f"old: {text_with_spaces}")
    for mistake, real in mistake_words.items():
        text_with_spaces = text_with_spaces.replace(mistake, real)
    print(f"new: {text_with_spaces}")
