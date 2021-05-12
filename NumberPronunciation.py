import random
import itertools
import functools


class Segment:
    def matches_any_feature_dict(self, options):
        # matches ANY dict, so if see a single match, return True immediately
        for option in options:
            if self.matches_feature_dict(option):
                return True
        # only if NO dict was matched, return False
        return False

    def matches_feature_dict(self, d):
        for k, v in d.items():
            try:
                self_attr = getattr(self, k)
            except AttributeError:
                # print(f"{self} has no attribute {k}")
                return False
            if self_attr != v:
                # print(f"{self}.{k} is {self_attr}, not {v}")
                return False
        # print(f"{self} matches {d}")
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
        if len(substr) < n:
            # substr was too short, constraint cannot apply
            return 0
        if word_matches_sequence(substr, self.sequence):
            # print(f"beginning substr {substr} violates markedness of {self.sequence}")
            return 1
        else:
            return 0

    def __repr__(self):
        return f"<Constraint *#{self.sequence}>"


class EndMarkednessConstraint:
    def __init__(self, sequence):
        self.sequence = sequence

    def get_violations(self, word):
        n = len(self.sequence)
        substr = word[-n:]
        if len(substr) < n:
            # substr was too short, constraint cannot apply
            return 0
        if word_matches_sequence(substr, self.sequence):
            # print(f"end substr {substr} violates markedness of {self.sequence}")
            return 1
        else:
            return 0

    def __repr__(self):
        return f"<Constraint *{self.sequence}#>"


class WholeWordMarkednessConstraint:
    def __init__(self, sequence):
        self.sequence = sequence

    def get_violations(self, word):
        n = len(self.sequence)
        if len(word) != n:
            return 0
        substr = word
        if word_matches_sequence(substr, self.sequence):
            # print(f"whole word {substr} violates markedness of {self.sequence}")
            return 1
        else:
            return 0

    def __repr__(self):
        return f"<Constraint *#{self.sequence}#>"



class AnywhereMarkednessConstraint:
    def __init__(self, sequence):
        self.sequence = sequence

    def get_violations(self, word):
        n = len(self.sequence)
        count = 0
        for i in range(len(word) - n + 1):
            substr = word[i:i+n]
            if len(substr) < n:
                # substr was too short, constraint cannot apply
                continue
            if word_matches_sequence(substr, self.sequence):
                # print(f"internal substr {substr} of word {word} violates markedness of {self.sequence} with badness {count}")
                count += 1
        return count

    def __repr__(self):
        return f"<Constraint *{self.sequence}>"


def word_matches_sequence(word, seq):
    assert len(word) == len(seq), f"can't align with different lengths: word {word} vs seq {seq}"
    # word must match ALL in seq, but each segment can match ANY of the corresponding options
    for w_seg, seq_seg_options in zip(word, seq):
        # print(f"checking if word seg {w_seg} matches options {seq_seg_options}")
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
        # print(f"evaluating candidates w.r.t. {constraint}")
        # print(f"candidates left: {candidates_left}")
        violations_by_candidate = {}
        for c in candidates_left:
            violations = constraint.get_violations(c)
            # print(f"candidate {c} has {violations} violations")
            violations_by_candidate[c] = violations
        min_violations = min(violations_by_candidate.values())
        candidates_left = [c for c,viols in violations_by_candidate.items() if viols == min_violations]
        candidates_removed = [c for c,viols in violations_by_candidate.items() if viols != min_violations]
        if len(candidates_left) == 1:
            # print(f"\nreturning optimal candidate: {candidates_left[0]}")
            # input("check")
            return candidates_left[0]
        elif len(candidates_left) == 0:
            raise RuntimeError("all candidates were rejected, problem in OT code")
        # print(f"\ndone with constraint {constraint}\nremoved: {candidates_removed}\nleft: {candidates_left}\n")
        # input("check")
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


def conjugate_number(n, CONSONANTS, VOWELS, CONSTRAINTS):
    print(f"root number chosen: {n}")
    prefixes = [""] + [str(x) for x in random.sample(list(range(100)), 3)]
    suffixes = [""] + [str(x) for x in random.sample(list(range(100)), 3)]

    for prefix in prefixes:
        for suffix in suffixes:
            this_n = prefix + str(n) + suffix  # DON'T cast to int because will lose leading zeros
            this_n_display = f"{prefix}-{n}-{suffix}"
            print_pronunciation(this_n, this_n_display, CONSONANTS, VOWELS, CONSTRAINTS)


def print_pronunciation(n_int, n_display_str, CONSONANTS, VOWELS, CONSTRAINTS):
    pronunciation = get_pronunciation_str(n_int, CONSONANTS, VOWELS, CONSTRAINTS)
    print(f"pronunciation of {n_display_str}: {s}")


def get_pronunciation_str(n_int, CONSONANTS, VOWELS, CONSTRAINTS):
    pronunciation = get_pronunciation(n_int, CONSONANTS, VOWELS, CONSTRAINTS)
    s = " ".join(x.symbol for x in pronunciation)  # join with space because some IPA symbols are wider than monospace
    return s


def print_pronunciations_of_numbers(ints, CONSONANTS, VOWELS, CONSTRAINTS):
    pronunciations = [get_pronunciation_str(n, CONSONANTS, VOWELS, CONSTRAINTS) for n in ints]
    s = " ; ".join(pronunciations)
    print(f"{ints} -> {s}")


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

    affricate_features = {"manner": "affricate"}
    approximant_features = {"manner": "approximant"}
    c_b_features = {"symbol": "b"}
    c_c_cedilla_features = {"symbol": "ç"}
    c_curly_j_features = {"symbol": "ʝ"}
    c_d_features = {"symbol": "d"}
    c_dzh_features = {"symbol": "ʤ"}
    c_eth_features = {"symbol": "ð"}
    c_f_features = {"symbol": "f"}
    c_features = {"cv": "C"}
    c_g_features = {"symbol": "g"}
    c_h_features = {"symbol": "h"}
    c_j_features = {"symbol": "j"}
    c_k_features = {"symbol": "k"}
    c_l_features = {"symbol": "l"}
    c_m_features = {"symbol": "m"}
    c_n_features = {"symbol": "n"}
    c_ng_features = {"symbol": "ŋ"}
    c_p_features = {"symbol": "p"}
    c_r_features = {"symbol": "r"}
    c_s_features = {"symbol": "s"}
    c_sh_features = {"symbol": "ʃ"}
    c_t_features = {"symbol": "t"}
    c_theta_features = {"symbol": "θ"}
    c_tsh_features = {"symbol": "ʧ"}
    c_v_features = {"symbol": "v"}
    c_w_features = {"symbol": "w"}
    c_z_features = {"symbol": "z"}
    c_zh_features = {"symbol": "ʒ"}
    dental_fricative_features = {"place": "dental", "manner": "fricative"}
    palatal_fricative_features = {"place": "palatal", "manner": "fricative"}
    postalveolar_fricative_features = {"place": "postalveolar", "manner": "fricative"}
    lateral_approximant_features = {"manner": "lateral_approximant"}
    nasal_features = {"manner": "nasal"}
    stop_features = {"manner": "stop"}
    v_a_features = {"symbol": "a"}
    v_ae_features = {"symbol": "æ"}
    v_e_features = {"symbol": "e"}
    v_eo_features = {"symbol": "ɤ"}
    v_eu_features = {"symbol": "ɯ"}
    v_features = {"cv": "V"}
    v_i_features = {"symbol": "i"}
    v_o_features = {"symbol": "o"}
    v_oe_features = {"symbol": "ø"}
    v_u_features = {"symbol": "u"}
    v_y_features = {"symbol": "y"}
    voiced_affricate_features = {"manner": "affricate", "voicing": "voiced"}
    voiced_features = {"voicing": "voiced"}
    voiced_fricative_features = {"manner": "fricative", "voicing": "voiced"}
    voiced_stop_features = {"manner": "stop", "voicing": "voiced"}
    voiceless_affricate_features = {"manner": "affricate", "voicing": "voiceless"}
    voiceless_features = {"voicing": "voiceless"}
    voiceless_stop_features = {"manner": "stop", "voicing": "voiceless"}

    B_seg = [c_b_features]
    CCEDILLA_seg = [c_c_cedilla_features]
    CURLYJ_seg = [c_curly_j_features]
    DZH_seg = [c_dzh_features]
    D_seg = [c_d_features]
    ETH_seg = [c_eth_features]
    F_seg = [c_f_features]
    G_seg = [c_g_features]
    H_seg = [c_h_features]
    I_seg = [v_i_features]
    J_seg = [c_j_features]
    K_seg = [c_k_features]
    L_seg = [c_l_features]
    M_seg = [c_m_features]
    NG_seg = [c_ng_features]
    N_seg = [c_n_features]
    P_seg = [c_p_features]
    R_seg = [c_r_features]
    SH_seg = [c_sh_features]
    S_seg = [c_s_features]
    THETA_seg = [c_theta_features]
    TSH_seg = [c_tsh_features]
    T_seg = [c_t_features]
    V_seg = [c_v_features]
    W_seg = [c_w_features]
    ZH_seg = [c_zh_features]
    Z_seg = [c_z_features]
    affricate_seg = [affricate_features]
    c_seg = [c_features]  # if it matches anything in the list, it will match the segment selector
    dental_fricative_seg = [dental_fricative_features]
    nasal_seg = [nasal_features]
    non_canonical_vowel_seg = [v_y_features, v_oe_features, v_ae_features, v_eu_features, v_eo_features]
    non_canonical_consonant_seg = [affricate_features, dental_fricative_features, postalveolar_fricative_features, palatal_fricative_features]
    optional_nasal_seg = [c_m_features, c_ng_features]
    sonorant_seg = [nasal_features, approximant_features, lateral_approximant_features]
    stop_seg = [stop_features]
    v_seg = [v_features]
    voiced_fricative_seg = [voiced_fricative_features]
    voiced_seg = [voiced_features]
    voiced_stop_or_affricate_seg = [voiced_stop_features, voiced_affricate_features]
    voiceless_seg = [voiceless_features]
    voiceless_stop_or_affricate_seg = [voiceless_stop_features, voiceless_affricate_features]
    voiceless_stop_seg = [voiceless_stop_features]

    B_seq = [B_seg]
    CCEDILLA_seq = [CCEDILLA_seg]
    CURLYJ_seq = [CURLYJ_seg]
    DZH_seq = [DZH_seg]
    D_seq = [D_seg]
    ETH_seq = [ETH_seg]
    F_seq = [F_seg]
    G_seq = [G_seg]
    H_seq = [H_seg]
    I_J_seq = [I_seg, J_seg]
    J_I_seq = [J_seg, I_seg]
    K_seq = [K_seg]
    L_seq = [L_seg]
    M_seq = [M_seg]
    NG_seq = [NG_seg]
    N_seq = [N_seg]
    P_seq = [P_seg]
    R_seq = [R_seg]
    SH_seq = [SH_seg]
    S_seq = [S_seg]
    THETA_seq = [THETA_seg]
    TSH_seq = [TSH_seg]
    T_seq = [T_seg]
    V_seq = [V_seg]
    W_seq = [W_seg]
    ZH_seq = [ZH_seg]
    Z_seq = [Z_seg]
    affricate_seq = [affricate_seg]
    c_seq = [c_seg]  # a sequence of one segment
    cc_seq = [c_seg, c_seg]  # this is a list of segments in order
    dental_fricative_seq = [dental_fricative_seg]
    non_canonical_vowel_seq = [non_canonical_vowel_seg]
    non_canonical_consonant_seq = [non_canonical_consonant_seg]
    optional_nasal_seq = [optional_nasal_seg]
    stop_seq = [stop_seg]
    v_nasal_v_seq = [v_seg, nasal_seg, v_seg]
    v_seq = [v_seg]
    v_stop_v_seq = [v_seg, stop_seg, v_seg]
    v_voiceless_v_seq = [v_seg, voiceless_seg, v_seg]
    voiced_fricative_seq = [voiced_fricative_seg]
    voiced_seq = [voiced_seg]
    voiced_stop_or_affricate_seq = [voiced_stop_or_affricate_seg]
    voiced_voiceless_seq = [voiced_seg, voiceless_seg]
    voiceless_seq = [voiceless_seg]
    voiceless_stop_or_affricate_seq = [voiceless_stop_or_affricate_seg]
    voiceless_stop_seq = [voiceless_stop_seg]
    voiceless_voiced_seq = [voiceless_seg, voiced_seg]
    vv_seq = [v_seg, v_seg]


    CONSTRAINTS = [
        WholeWordMarkednessConstraint(c_seq),
        # BeginningMarkednessConstraint(cc_seq),
        # BeginningMarkednessConstraint(vv_seq),
        AnywhereMarkednessConstraint(cc_seq),
        AnywhereMarkednessConstraint(vv_seq),
        AnywhereMarkednessConstraint(voiceless_voiced_seq),
        AnywhereMarkednessConstraint(voiced_voiceless_seq),

        AnywhereMarkednessConstraint(non_canonical_vowel_seq),
        BeginningMarkednessConstraint(optional_nasal_seq),
        BeginningMarkednessConstraint(voiceless_stop_or_affricate_seq),
        BeginningMarkednessConstraint(voiced_fricative_seq),
        BeginningMarkednessConstraint(R_seq),
        AnywhereMarkednessConstraint(v_voiceless_v_seq),
        AnywhereMarkednessConstraint(v_nasal_v_seq),
        AnywhereMarkednessConstraint(v_stop_v_seq),
        EndMarkednessConstraint(dental_fricative_seq),
        AnywhereMarkednessConstraint(W_seq),
        AnywhereMarkednessConstraint(H_seq),
        EndMarkednessConstraint(stop_seq),

        AnywhereMarkednessConstraint(J_I_seq),
        EndMarkednessConstraint(I_J_seq),
        AnywhereMarkednessConstraint(CURLYJ_seq),
        AnywhereMarkednessConstraint(CCEDILLA_seq),
        AnywhereMarkednessConstraint(non_canonical_consonant_seq),

        EndMarkednessConstraint(voiced_seq),
        BeginningMarkednessConstraint(v_seq),
        # TODO add "goodness" constraints that encourage double consonants e.g. ll, nn, tt, kk


        # low-ranked stuff that I'm not actually using for my ordering, but which can be used in random orderings
        AnywhereMarkednessConstraint(B_seq),
        AnywhereMarkednessConstraint(D_seq),
        AnywhereMarkednessConstraint(DZH_seq),
        AnywhereMarkednessConstraint(ETH_seq),
        AnywhereMarkednessConstraint(F_seq),
        AnywhereMarkednessConstraint(G_seq),
        AnywhereMarkednessConstraint(K_seq),
        AnywhereMarkednessConstraint(L_seq),
        AnywhereMarkednessConstraint(M_seq),
        AnywhereMarkednessConstraint(N_seq),
        AnywhereMarkednessConstraint(NG_seq),
        AnywhereMarkednessConstraint(P_seq),
        AnywhereMarkednessConstraint(S_seq),
        AnywhereMarkednessConstraint(SH_seq),
        AnywhereMarkednessConstraint(T_seq),
        AnywhereMarkednessConstraint(THETA_seq),
        AnywhereMarkednessConstraint(TSH_seq),
        AnywhereMarkednessConstraint(V_seq),
        AnywhereMarkednessConstraint(Z_seq),
        AnywhereMarkednessConstraint(ZH_seq),

    ]

    # random.shuffle(CONSTRAINTS)

    mode = input("Select mode:\n1. conjugate random number to observe stem changes\n2. pronounce number(s) from input\n")
    if mode == "1":
        n = random.randrange(10, 10000)
        n = str(n)[1:]  # so leading zeros may also be represented
        conjugate_number(n, CONSONANTS, VOWELS, CONSTRAINTS)
    elif mode == "2":
        num_str = input("Enter number(s) (separated by spaces): ")
        nums = [int(x) for x in num_str.split()]
        print_pronunciations_of_numbers(nums, CONSONANTS, VOWELS, CONSTRAINTS)
    else:
        print("unknown mode")

    # for n in range(1000):
    #     print_pronunciation(n, str(n), CONSONANTS, VOWELS, CONSTRAINTS)

