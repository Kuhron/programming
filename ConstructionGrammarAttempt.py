import random

constructions = {
    # "{AtTime}": "{TimeNP} lum",  # orig
    "{AtTime}": "luma {TimeNP}",
    "{AtPlace}": "{PlaceNP} lan",
    # "{ItSeemsLike}": "lirlattl mall {S}",  # orig
    "{ItSeemsLike}": "lirlattl {S} mallu",
    "{XNeedsTo}": "aunn{SubjectSuffix} {VPInf}",
    "{VPInf}": ["{VtInfObj}", "{ViInf}"],
    "{VtInfObj}": "{VtInf} {Obj}",
    "{Obj}": ["{NP}", "{NPost}"],
    # "{NPost}": "{NP} {PostP}",  # orig
    "{NPost}": "{PostP} ollan {NP}",
    "{PostP}": ["{AtTime}", "{AtPlace}"],
    "{S}": ["{XNeedsTo}", "{PersonS}", "{VPPronInfl}"],
    # "{PersonS}": "{PersonNP} {VPInfl}",  # orig
    "{PersonS}": "{VPInfl} a, {PersonNP} ta",
    "{VPInfl}": ["{VtInfl}", "{ViInfl}"],
    "{VtInfl}": "{VtInf}arl {Obj}",
    "{ViInfl}": "{ViInf}arl",
    "{VPPronInfl}": ["{VtPronInfl}", "{ViPronInfl}"],
    "{VtPronInfl}": "{VtInf}{SubjectSuffix} {Obj}",
    "{ViPronInfl}": "{ViInf}{SubjectSuffix}",
    "{TimeNP}": ["nalu", "merlu", "llasu"],
    "{PlaceNP}": ["rlu", "Tahelle", "ttla", "lelle"],
    "{PersonNP}": ["narla", "rlarella", "rlamarla", "rlamerella", "Surlan"],
    "{ThingNP}": ["amalo", "awll", "zlettlo", "nllde"],
    "{NP}": ["{TimeNP}", "{PlaceNP}", "{PersonNP}", "{ThingNP}"],
    "{SubjectSuffix}": ["al", "el", "arl", "aral", "arel", "ararl"],
    "{VtInf}": ["llrl", "rar", "trlax", "mllm", "nol"],
    "{ViInf}": ["llr", "rar", "delm", "mllm", "uall", "llaw"],
    "{NNPoss}": "{NPPossessed} {NPPossessor} ffra",
    "{NPPossessed}": "{NP}",
    "{NPPossessor}": "{NP}",
}


def fill_construction_random(cx, psrs):
    while "{" in cx or "}" in cx:
        sub_cx_label = cx.split("{")[1].split("}")[0]
        sub_cx_label = "{" + sub_cx_label + "}"
        psr_maps_to = psrs[sub_cx_label]
        if type(psr_maps_to) is str:
            # it's a template
            sub_cx = fill_construction_random(psr_maps_to, psrs)
        elif type(psr_maps_to) is list:
            # it's either a list of other cx labels or a list of lexical items
            sub_cx = random.choice(psr_maps_to)
        else:
            raise TypeError
        cx = cx.replace(sub_cx_label, sub_cx)
        # print("current cx: {}".format(cx))
    return cx


def translate(d, constructions):
    # recursively fill in the slots by name from the dict keys
    assert len(d) == 1, "must only have one top-level construction type for translation"
    k = list(d.keys())[0]
    v = d[k]
    top_level_container = constructions[k]
    return translate_helper(k, v, top_level_container, constructions)
    

def translate_helper(k, v, container, constructions):
    print("called translate_helper with k={}, v={}, container={}".format(k, v, container))
    if type(container) is list:
        # it is either a list of more labels or a list of lexical items
        if type(v) is dict:
            assert len(v) == 1
            selected_v = list(v.keys())[0]
        else:
            selected_v = v
        assert selected_v in container
        # the passed v is the new container, select just that one for future replacement
        container = selected_v
        return translate_helper(k, v, container, constructions)

    if type(v) is str:
        assert v in constructions[k]
        if v in constructions:
            # it's the label of another cx, call the func again
            sub_result = translate_helper(v, constructions[v], container, constructions)
            container = sub_result
        else:
            # it's a lexical item
            container = container.replace(k, v)
    elif type(v) is dict:
        # will need to recurse
        for sub_k, sub_v in v.items():
            sub_container = constructions[sub_k]
            sub_result = translate_helper(sub_k, sub_v, sub_container, constructions)
            container = container.replace(sub_k, sub_result)
    else:
        raise TypeError("value of type {}".format(type(v)))
    return container


if __name__ == "__main__":
    for _ in range(100):
        sentence = fill_construction_random("{S}", constructions)
        print(sentence)
    print("----\n")

    to_translate = {"{S}": {
        "{PersonS}": {
            "{PersonNP}": "Surlan",
            "{VPInfl}": {
                "{VtInfl}": {
                    "{VtInf}": "trlax",
                    "{Obj}": {
                        "{NPost}": {
                            "{NP}": {
                                "{PersonNP}": "rlarella",
                            },
                            "{PostP}": {
                                "{AtTime}": {
                                    "{TimeNP}": "nalu",
                                },
                            },
                        },
                    },
                },
            },
        },
    }}

    # to_translate = {"{PersonNP}": "Surlan"}  # base case
    # to_translate = {"{AtTime}": {
    #     "{TimeNP}": "nalu",
    # }}
    # to_translate = {"{NNPoss}": {
    #     "{NPPossessed}": {
    #         "{NP}": {
    #             "{ThingNP}": "zlettlo",
    #         },
    #     },
    #     "{NPPossessor}": {
    #         "{NP}": {
    #             "{PersonNP}": "narla",
    #         },
    #     },
    # }}

    sentence = translate(to_translate, constructions)
    print(sentence)
