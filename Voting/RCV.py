import sys
import math


def get_rankings(fp):
    with open(fp) as f:
        lines = f.readlines()
    lines = [l.strip().split(",") for l in lines]
    header = lines[0]
    assert header[0] == "response", f"bad header: {header!r}"
    assert all(header[i] == str(i) for i in range(1, len(header))), f"bad header: {header!r}"
    response_ids = set()
    candidates = set()
    rankings = {}
    for row in lines[1:]:
        response_id, *ranking = row
        if response_id in response_ids:
            raise ValueError(f"repeated {response_id = !r}")
        if candidates == set():
            candidates = set(ranking)
        if set(ranking) != candidates:
            raise ValueError(f"expected candidates {sorted(candidates)} but got {sorted(ranking)}")
        rankings[response_id] = ranking

    return rankings


def get_all_candidates(rankings):
    s = set()
    for ranking in rankings.values():
        s |= set(ranking)
    return s


def get_first_choice_votes(rankings):
    # once we've started eliminating candidates and call this again, we need to make sure all of them are in the dict
    d = {x: 0 for x in get_all_candidates(rankings)}
    for ranking in rankings.values():
        c = ranking[0]
        d[c] += 1
    return d


def remove_losers_from_rankings(losers, rankings):
    return {response_id: [c for c in ranking if c not in losers] for response_id, ranking in rankings.items()}


def get_winner_from_rankings(rankings):
    # have a few algorithms in order for tiebreaking

    w = get_winner_from_rankings_instant_runoff(rankings)
    if type(w) is list:
        w = get_winner_from_rankings_borda(rankings, eligible=w)

    return w


def get_winner_from_rankings_borda(rankings, eligible=None):
    n = None
    if eligible is None:
        counts = None
    else:
        counts = {c: 0 for c in eligible}

    for ranking in rankings.values():
        if n is None:
            n = len(ranking)
        if counts is None:
            counts = {c: 0 for c in ranking}
        for i, c in enumerate(ranking):
            if c not in counts:
                continue
            n_below = n - 1 - i
            counts[c] += n_below
    print(f"Borda counts: {counts}")
    winners = [c for c,n in counts.items() if n == max(counts.values())]
    print(f"Borda winners: {winners}")
    if len(winners) == 1:
        return winners[0]
    else:
        raise Exception("no winner!")


def get_winner_from_rankings_instant_runoff(rankings):
    # https://ballotpedia.org/Ranked-choice_voting_(RCV)
    print("\n---- new round ----")
    for response_id, ranking in rankings.items():
        print(f"response {response_id}: " + ", ".join(f"{i+1}:{c}" for i,c in enumerate(ranking)) )

    # if a candidate wins an outright majority (>50%) of first-choice votes, it wins
    first_choice_votes_by_candidate = get_first_choice_votes(rankings)
    print(f"{first_choice_votes_by_candidate = }")
    majority_threshold = math.floor(len(rankings) / 2 + 1)
    print(f"{majority_threshold = }")
    assert type(majority_threshold) is int
    majs = [c for c,n in first_choice_votes_by_candidate.items() if n >= majority_threshold]
    assert len(majs) <= 1, f"impossible to have more than one majority winner, but got first-choice vote counts: {first_choice_votes_by_candidate}"
    if len(majs) == 1:
        return majs[0]

    # otherwise, the candidate with the fewest first-preference votes is eliminated
    # and all the first-preference votes for that candidate are eliminated
    # and the rankings by the voters who had it as first choice are updated (moved up by one rank) since they lost their first choice

    losers = [c for c,n in first_choice_votes_by_candidate.items() if n == min(first_choice_votes_by_candidate.values())]
    print(f"{losers = }")
    assert len(losers) >= 1, f"should have at least one candidate with min votes, but got counts: {first_choice_votes_by_candidate}"

    # if we would remove all remaining candidates, then we have a tie
    if len(losers) == len(first_choice_votes_by_candidate):
        return losers  # for use in tiebreaking, we know who almost won here

    rankings = remove_losers_from_rankings(losers, rankings)

    # this repeats until a candidate wins an outright majority of first-choice votes

    return get_winner_from_rankings_instant_runoff(rankings)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python RCV.py [input_csv_fp]")
        sys.exit()

    fp = sys.argv[1]
    print(f"analyzing {fp}")

    rankings = get_rankings(fp)
    winner = get_winner_from_rankings(rankings)
    print(f"the winner is: {winner!r}")

