import random
import itertools


class Segment:
    def matches_any_feature_dict(self, options):
        for option in options:
            if not self.matches_feature_dict(option):
                return False
        return True

    def matches_feature_dict(self, d):
        for k, v in d.items():
            try:
                self_attr = getattr(self, k)
            except AttributeError:
                return False
            if self_attr != v:
                return False
        return True

    def __repr__(self):
        return self.symbol


class Consonant(Segment):
    def __init__(self, symbol, voicing, place, manner):
        self.symbol = symbol
        self.place = place
        self.manner = manner
        self.voicing = voicing
        self.cv = "C"


class Vowel(Segment):
    def __init__(self, symbol):
        self.symbol = symbol
        self.cv = "V"


class BeginningMarkednessConstraint:
    def __init__(self, sequence):
        self.sequence = sequence

    def get_violations(self, word):
        n = len(self.sequence)
        substr = word[:n]
        if word_matches_sequence(substr, self.sequence):
            return 1
        else:
            return 0


class EndMarkednessConstraint:
    def __init__(self, sequence):
        self.sequence = sequence

    def get_violations(self, word):
        n = len(self.sequence)
        substr = word[-n:]
        if word_matches_sequence(substr, self.sequence):
            return 1
        else:
            return 0


class AnywhereMarkednessConstraint:
    def __init__(self, sequence):
        self.sequence = sequence

    def get_violations(self, word):
        n = len(self.sequence)
        count = 0
        for i in range(len(word) - n + 1):
            substr = word[i:i+n]
            if word_matches_sequence(substr, self.sequence):
                count += 1
        return count


def word_matches_sequence(word, seq):
    assert len(word) == len(seq), "can't align with different lengths"
    for w_seg, seq_seg_options in zip(word, seq):
        if w_seg.matches_any_feature_dict(seq_seg_options):
            # it matches so far, continue
            continue
        else:
            return False
    return True


def get_pronunciation(n, CONSONANTS, VOWELS, CONSTRAINTS):
    candidates = get_candidates(n, CONSONANTS, VOWELS)
    optimal = get_optimal_candidate(candidates, CONSTRAINTS)
    return optimal


def get_candidates(n, CONSONANTS, VOWELS):
    s = str(n)
    digits = [int(x) for x in s]
    segment_options = [get_segment_options(digit, CONSONANTS, VOWELS) for digit in digits]
    return itertools.product(*segment_options)


def get_segment_options(n, CONSONANTS, VOWELS):
    consonant_options = CONSONANTS[n]
    vowel_options = VOWELS[n]
    return consonant_options + vowel_options


def get_optimal_candidate(candidates, CONSTRAINTS):
    # strict ranking
    candidates_left = [x for x in candidates]
    for constraint in CONSTRAINTS:
        # print(f"candidates left: {candidates_left}")
        violations_by_candidate = {}
        for c in candidates_left:
            violations = constraint.get_violations(c)
            # print(f"candidate {c} has {violations} violations")
            violations_by_candidate[c] = violations
        min_violations = min(violations_by_candidate.values())
        candidates_left = [c for c,viols in violations_by_candidate.items() if viols == min_violations]
        if len(candidates_left) == 1:
            return candidates_left[0]
        elif len(candidates_left) == 0:
            raise RuntimeError("all candidates were rejected, problem in OT code")
    # if got here, we are out of constraints but must have more than one candidate left
    raise Exception(f"more than one candidate passed: {candidates_left}")


def get_segment_cv(seg):
    if type(seg) is Consonant:
        return "C"
    elif type(seg) is Vowel:
        return "V"
    else:
        raise ValueError(seg)


def get_word_cv(w):
    return "".join(get_segment_cv(seg) for seg in w)


if __name__ == "__main__":
    c_sh = Consonant("ʃ", voicing="voiceless", place="postalveolar", manner="fricative")
    c_zh = Consonant("ʒ", voicing="voiced", place="postalveolar", manner="fricative")
    c_j = Consonant("j", voicing="voiced", place="palatal", manner="approximant")
    c_curly_j = Consonant("ʝ", voicing="voiced", place="palatal", manner="fricative")
    c_c_cedilla = Consonant("ç", voicing="voiceless", place="palatal", manner="fricative")
    c_d = Consonant("d", "voiced", "alveolar", "stop")
    c_r = Consonant("r", "voiced", "alveolar", "approximant")
    c_t = Consonant("t", "voiceless", "alveolar", "stop")
    c_theta = Consonant("θ", "voiceless", "dental", "fricative")
    c_eth = Consonant("ð", "voiced", "dental", "fricative")
    c_tsh = Consonant("ʧ", "voiceless", "postalveolar", "affricate")
    c_dzh = Consonant("ʤ", "voiced", "postalveolar", "affricate")
    c_f = Consonant("f", "voiceless", "labiodental", "fricative")
    c_v = Consonant("v", "voiced", "labiodental", "fricative")
    c_h = Consonant("h", "voiceless", "glottal", "fricative")
    c_s = Consonant("s", "voiceless", "alveolar", "fricative")
    c_z = Consonant("z", "voiced", "alveolar", "fricative")
    c_w = Consonant("w", "voiced", "labiovelar", "approximant")
    c_g = Consonant("g", "voiced", "velar", "stop")
    c_k = Consonant("k", "voiceless", "velar", "stop")
    c_ng = Consonant("ŋ", "voiced", "velar", "nasal")
    c_l = Consonant("l", "voiced", "alveolar", "lateral_approximant")
    c_b = Consonant("b", "voiced", "bilabial", "stop")
    c_p = Consonant("p", "voiceless", "bilabial", "stop")
    c_m = Consonant("m", "voiced", "bilabial", "nasal")
    c_n = Consonant("n", "voiced", "alveolar", "nasal")
    CONSONANTS = [
        [c_sh, c_zh],
        [c_j, c_curly_j, c_c_cedilla],
        [c_d, c_r, c_t],
        [c_theta, c_eth, c_tsh, c_dzh],
        [c_f, c_v, c_h],
        [c_s, c_z, c_w],
        [c_g, c_k, c_ng],
        [c_l],
        [c_b, c_p, c_m],
        [c_n],
    ]
    VOWELS = [[Vowel(x)] for x in "oiyeauɤɯøæ"]

    c_features = {"cv": "C"}
    v_features = {"cv": "V"}
    stop_features = {"manner": "stop"}
    nasal_features = {"manner": "nasal"}
    voiceless_features = {"voicing": "voiceless"}
    voiced_features = {"voicing": "voiced"}
    v_y_features = {"symbol": "y"}
    v_oe_features = {"symbol": "ø"}
    v_ae_features = {"symbol": "æ"}
    v_eu_features = {"symbol": "ɯ"}
    v_eo_features = {"symbol": "ɤ"}
    c_m_features = {"symbol": "m"}
    c_ng_features = {"symbol": "ŋ"}
    dental_fricative_features = {"place": "dental", "manner": "fricative"}
    c_w_features = {"symbol": "w"}
    c_h_features = {"symbol": "h"}
    voiced_fricative_features = {"manner": "fricative", "voicing": "voiced"}
    c_j_features = {"symbol": "j"}
    v_i_features = {"symbol": "i"}
    v_e_features = {"symbol": "e"}
    v_a_features = {"symbol": "a"}
    v_o_features = {"symbol": "o"}
    v_u_features = {"symbol": "u"}
    c_c_cedilla_features = {"symbol": "ç"}

    c_seg = [c_features]  # if it matches anything in the list, it will match the segment selector
    v_seg = [v_features]
    stop_seg = [stop_features]
    nasal_seg = [nasal_features]
    voiced_seg = [voiced_features]
    voiceless_seg = [voiceless_features]
    non_canonical_vowel_seg = [v_y_features, v_oe_features, v_ae_features, v_eu_features, v_eo_features]
    optional_nasal_seg = [c_m_features, c_ng_features]
    dental_fricative_seg = [dental_fricative_features]
    W_seg = [c_w_features]
    H_seg = [c_h_features]
    voiced_fricative_seg = [voiced_fricative_features]
    CCEDILLA_seg = [c_c_cedilla_features]
    J_seg = [c_j_features]
    I_seg = [v_i_features]

    c_seq = [c_seg]  # a sequence of one segment
    v_seq = [v_seg]
    cc_seq = [c_seg, c_seg]  # this is a list of segments in order
    vv_seq = [v_seg, v_seg]
    v_voiceless_v_seq = [v_seg, voiceless_seg, v_seg]
    stop_seq = [stop_seg]
    non_canonical_vowel_seq = [non_canonical_vowel_seg]
    optional_nasal_seq = [optional_nasal_seg]
    v_stop_v_seq = [v_seg, stop_seg, v_seg]
    dental_fricative_seq = [dental_fricative_seg]
    W_seq = [W_seg]
    H_seq = [H_seg]
    voiced_fricative_seq = [voiced_fricative_seg]
    CCEDILLA_seq = [CCEDILLA_seg]
    J_I_seq = [J_seg, I_seg]

    CONSTRAINTS = [
        AnywhereMarkednessConstraint(non_canonical_vowel_seq),
        BeginningMarkednessConstraint(optional_nasal_seq),
        BeginningMarkednessConstraint(voiced_fricative_seq),
        AnywhereMarkednessConstraint(cc_seq),
        AnywhereMarkednessConstraint(vv_seq),
        AnywhereMarkednessConstraint(v_voiceless_v_seq),
        AnywhereMarkednessConstraint(v_stop_v_seq),
        AnywhereMarkednessConstraint(dental_fricative_seq),
        AnywhereMarkednessConstraint(W_seq),
        AnywhereMarkednessConstraint(H_seq),
        EndMarkednessConstraint(stop_seq),

        AnywhereMarkednessConstraint(J_I_seq),
        AnywhereMarkednessConstraint(CCEDILLA_seq),

        BeginningMarkednessConstraint(v_seq),
    ]

    n = random.randint(0, 9999)
    print(f"number chosen: {n}")
    pronunciation = get_pronunciation(n, CONSONANTS, VOWELS, CONSTRAINTS)
    s = "".join(x.symbol for x in pronunciation)
    print(f"pronunciation: {s}")
