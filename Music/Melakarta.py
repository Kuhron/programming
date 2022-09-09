# chart of Melakarta ragas
# also with Western notation and my Greek/color notation for helping me memorize them

import csv


GREEK_INTERVAL_TO_SEMITONES = {
    "ω": 0, "α": 1, "β": 2, "γ": 3, "δ": 4, "ε": 5, "ζ": 6,
    "η": 7, "θ": 8, "ι": 9, "κ": 10, "λ": 11, "μ" : 12
}
SWARA_INTERVAL_TO_SEMITONES = {
    "S": 0, "R1": 1, "R2": 2, "G1": 2, "R3": 3, "G2": 3, "G3": 4, "M1": 5, "M2": 6,
    "P": 7, "D1": 8, "D2": 9, "N1": 9, "D3": 10, "N2": 10, "N3": 11, "S'": 12
}
SEMITONES_TRIPLE_TO_GREEK = {
    (1,2,1): "ν", (1,2,2): "ξ", (1,2,3): "o", (1,3,1): "π", (1,3,2): "ρ",
    (2,1,2): "σ", (2,1,3): "τ", (2,2,1): "υ", (2,2,2): "φ", (3,1,2): "χ", (3,1,1): "ψ",
    (1,1,3): "?", (1,1,4): "?",
}
GREEK_TRIPLE_TO_COLOR = {
    "ν": "gold", "ξ": "red", "o": "orange", "π": "yellow", "ρ": "green",
    "σ": "blue", "τ": "purple", "υ": "white", "φ": "silver", "χ": "black", "ψ": "pink",
    "?": "?",
}



def get_greek_from_swara(swara):
    st = SWARA_INTERVAL_TO_SEMITONES[swara]
    g, = [c for c,s in GREEK_INTERVAL_TO_SEMITONES.items() if s == st]
    return g


def get_greeks_from_swaras(swaras):
    return [get_greek_from_swara(x) for x in swaras]


def get_semitones_cumulative_from_swaras(swaras):
    return [SWARA_INTERVAL_TO_SEMITONES[x] for x in swaras]


def get_semitones_steps_from_swaras(swaras):
    stc = get_semitones_cumulative_from_swaras(swaras)
    res = []
    for i in range(1, len(stc)):
        st1, st0 = stc[i], stc[i-1]
        res.append(st1 - st0)
    return res


def get_madhyama_from_swaras(swaras):
    # based on the ma value
    ma = swaras[3]
    assert ma.startswith("M") and len(ma) == 2, ma
    n = ma[-1]
    return {"1": "shuddha", "2": "prati"}[n]


def get_chakra_from_swaras(swaras):
    # based on ri and ga values
    ri = swaras[1]
    ga = swaras[2]
    assert ri.startswith("R") and len(ri) == 2, ri
    assert ga.startswith("G") and len(ga) == 2, ga
    nr = int(ri[-1])
    ng = int(ga[-1])
    madhyama = ["shuddha", "prati"].index(get_madhyama_from_swaras(swaras))
    return {
        (1, 1): ["indu", "rishi"],
        (1, 2): ["netra", "vasu"],
        (1, 3): ["agni", "brahma"],
        (2, 2): ["veda", "disi"],
        (2, 3): ["bana", "rudra"],
        (3, 3): ["rutu", "aditya"],
    }[(nr, ng)][madhyama]


def get_greeks_halves_from_semitones_steps(semitones_steps):
    # first four notes and last four (including last sa) determine this, the ma-pa step isn't used because pa is always in the same place
    assert len(semitones_steps) == 7, semitones_steps
    first_four = semitones_steps[:3]
    last_four = semitones_steps[-3:]
    return (get_greek_half_from_semitones_steps(first_four), get_greek_half_from_semitones_steps(last_four))


def get_greek_half_from_semitones_steps(steps):
    assert len(steps) == 3, steps
    return SEMITONES_TRIPLE_TO_GREEK[tuple(steps)]


def get_colors_from_greeks_halves(greeks_halves):
    return [GREEK_TRIPLE_TO_COLOR[x] for x in greeks_halves]


def load_ragas():
    fp = "ragas.csv"
    res = []
    with open(fp) as f:
        for row in csv.DictReader(f):
            number = int(row["number"])
            name = row["name"]
            while name.endswith(" "):
                name = name[:-1]
            swaras = row["swaras"].split(" ")
            if "" in swaras:
                swaras.remove("")
            raga = Raga(number, name, swaras)
            res.append(raga)
    return res


class Raga:
    def __init__(self, number, name, swaras):
        self.number = number
        self.name = name
        self.swaras = swaras
        self.madhyama = get_madhyama_from_swaras(swaras)
        self.chakra = get_chakra_from_swaras(swaras)
        self.greeks = get_greeks_from_swaras(swaras)
        self.semitones_cumulative = get_semitones_cumulative_from_swaras(swaras)
        self.semitones_steps = get_semitones_steps_from_swaras(swaras)
        self.greeks_halves = get_greeks_halves_from_semitones_steps(self.semitones_steps)
        self.colors = get_colors_from_greeks_halves(self.greeks_halves)

    def get_notes(self, tonic, wkj_note_names=True):
        if wkj_note_names:
            note_names = ["C", "K", "D", "H", "E", "F", "X", "G", "J", "A", "R", "B"]
        else:
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        tonic_index = note_names.index(tonic)
        return [note_names[(tonic_index + i) % 12] for i in self.semitones_cumulative]

    def __repr__(self):
        swaras = ", ".join(self.swaras)
        greeks = ", ".join(self.greeks)
        return f"<Raga #{self.number} {self.name}, {self.madhyama} madhyama, {self.chakra} chakra,\n\t{swaras}\n\t{greeks}\n\t{self.greeks_halves} = {self.colors}>"



if __name__ == "__main__":
    ragas = load_ragas()
    for raga in ragas:
        print(raga)
        # for tonic in "CKDHEFXGJARB":
        tonic = "C"
        print(f"with tonic of {tonic}:", ", ".join(raga.get_notes(tonic)))
        print()
