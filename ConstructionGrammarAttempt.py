import random

constructions = {
    "AtTime": ["{TimeNP} lum"],
    "AtPlace": ["{PlaceNP} lan"],
    "ItSeemsLike": ["lirlattl mall {S}"],
    "XNeedsTo": ["aunn{SubjectSuffix} {VPInf}"],
    "VPInf": ["{VtInf} {Obj}", "{ViInf}"],
    "Obj": ["{NP}", "{NP} {PostP}"],
    "PostP": ["{AtTime}", "{AtPlace}"],
    "S": ["{XNeedsTo}", "{PersonNP} {VPInfl}", "{VPPronInfl}"],
    "VPInfl": ["{VtInf}arl {Obj}", "{ViInf}arl"],
    "VPPronInfl": ["{VtInf}{SubjectSuffix} {Obj}", "{ViInf}{SubjectSuffix}"],
    "TimeNP": ["nalu", "merlu", "llasu"],
    "PlaceNP": ["rlu", "Tahelle", "ttla", "lelle"],
    "PersonNP": ["narla", "rlarella", "rlamarla", "rlamerella", "Surlan"],
    "NP": ["{TimeNP}", "{PlaceNP}", "{PersonNP}"],
    "SubjectSuffix": ["al", "el", "arl", "aral", "arel", "ararl"],
    "VtInf": ["llrl", "rar", "trlax", "mllm", "nol"],
    "ViInf": ["llr", "rar", "delm", "mllm", "uall", "llaw"],
}


def fill_construction(cx, psrs):
    while "{" in cx or "}" in cx:
        sub_cx_label = cx.split("{")[1].split("}")[0]
        sub_cx = random.choice(psrs[sub_cx_label])
        cx = cx.replace("{"+sub_cx_label+"}", sub_cx)
        # print("current cx: {}".format(cx))
    return cx


if __name__ == "__main__":
    for _ in range(100):
        sentence = fill_construction("{S}", constructions)
        print(sentence)
