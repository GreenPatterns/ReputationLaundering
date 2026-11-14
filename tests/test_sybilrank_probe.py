"""Tests for keeper/analysis/sybilrank_probe.py: SybilRank as a tier source in the admission probe (CPU only)."""
import json
import os

import numpy as np
import pytest

from keeper.analysis import sybilrank_probe as SR
from tests.credibility_common import graph

# honest toy graph: 0 is the seed (endorsed by 1 and 2 in the seed window); 0, 3 and 4 endorse one another in a
# triangle, so the undirected graph is not bipartite and every account holds trust after a few steps
HONEST = [(1, 0, +1, 0), (2, 0, +1, 1), (0, 3, +1, 5), (3, 4, +1, 6), (4, 0, +1, 7)]
K_TOY = 6


def test_trust_follows_the_hand_computed_power_iteration():
    # Arrange: path 0 - 1 - 2 with the seed at one end; degrees 1, 2, 1
    endorse = np.array([[0, 1], [1, 2]])

    # Act
    history = SR.sybilrank_trust(3, endorse, seeds=[0], K=2, return_history=True)

    # Assert: all trust moves to the middle, then splits evenly
    assert [h.tolist() for h in history] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]


def test_endorsement_direction_and_mutual_pairs_give_one_undirected_edge():
    one_way = SR.sybilrank_trust(2, np.array([[0, 1]]), seeds=[0], K=1)
    mutual = SR.sybilrank_trust(2, np.array([[0, 1], [1, 0]]), seeds=[0], K=1)

    assert one_way.tolist() == mutual.tolist() == [0.0, 1.0]


def test_trust_is_conserved_and_an_isolated_seed_keeps_its_share():
    endorse = np.array([[0, 1], [1, 2], [2, 0], [2, 3]])

    trust = SR.sybilrank_trust(5, endorse, seeds=[0, 4], K=7)

    assert trust.sum() == pytest.approx(1.0) and trust[4] == pytest.approx(0.5)


def test_score_is_trust_over_degree_and_zero_without_neighbours():
    endorse = np.array([[0, 1], [1, 2]])

    score = SR.normalised_score(np.array([0.2, 0.6, 0.2, 0.5]), SR.degrees(4, endorse))

    assert score.tolist() == pytest.approx([0.2, 0.3, 0.2, 0.0])


def test_admission_needs_the_threshold_and_an_endorsement_from_a_trusted_rater():
    Tr = graph(HONEST)
    score = SR.scores(Tr, 5, seeds=[0], K=K_TOY)

    all_in = SR.admitted(Tr, 5, score, theta=0.0)
    none_in = SR.admitted(Tr, 5, score, theta=score.max() + 1.0)

    assert not none_in.any()
    assert all_in[[0, 3, 4]].all()          # each holds a positive rating from an account above the threshold
    assert all_in.sum() == 3                # accounts 1 and 2 hold no incoming rating, so they stay out


def test_threshold_is_the_largest_one_that_keeps_the_target_share():
    Tr = graph(HONEST)
    credible = np.array([True, False, False, True, True])
    score = SR.scores(Tr, 5, seeds=[0], K=K_TOY)

    theta, kept = SR.calibrate_theta(Tr, 5, score, credible, target=2 / 3)

    assert kept >= 2 / 3
    higher = np.unique(score[score > theta])         # every account's score is a candidate
    for candidate in higher:                # every larger candidate keeps too few
        assert (SR.admitted(Tr, 5, score, candidate) & credible).sum() / credible.sum() < 2 / 3


def test_fakes_without_an_attack_edge_are_never_admitted():
    Tr = graph(HONEST)
    rows = SR.region_rows([], [5, 6, 7], "ring")

    case = SR.region_case(Tr, 5, rows, n_fakes=3, snap=9, seeds=[0], K=K_TOY, theta=1e-12)

    assert case["admitted"] == 0 and case["region_trust"] == 0.0


def test_one_attack_edge_admits_fakes_only_while_their_score_reaches_the_threshold():
    Tr = graph(HONEST)
    rows = SR.region_rows([3], [5, 6, 7], "star")

    loose = SR.region_case(Tr, 5, rows, n_fakes=3, snap=9, seeds=[0], K=K_TOY, theta=1e-12)
    strict = SR.region_case(Tr, 5, rows, n_fakes=3, snap=9, seeds=[0], K=K_TOY, theta=10.0)

    assert loose["admitted"] == 3 and strict["admitted"] == 0
    assert 0.0 < loose["region_trust"] < 1.0


def test_the_paper_quotes_the_stored_probe_result():
    """RQ1's SybilRank sentence must match results/analysis/sybilrank_probe.json (matched retention)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "results", "analysis", "sybilrank_probe.json")) as f:
        networks = json.load(f)["networks"]
    with open(os.path.join(root, "paper", "main_scem.tex")) as f:
        paper = f.read()
    blocks = [networks[ds] for ds in ("bitcoinalpha", "bitcoinotc", "wiki-rfa")]
    sybilrank = [b["sybilrank"]["probe"]["matched"] for b in blocks]

    def listed(values):
        return "${}$, ${}$ and ${}$".format(*values)

    rule = [b["sybilrank"]["probe"]["rule"] for b in blocks]
    seed_share = [100.0 * b["n_seeds"] / b["n_ref"] for b in blocks]

    assert all(c["met"] for b in blocks for c in b["sybilrank"]["calibrations"].values())
    assert not any(p["lower_bound"] for p in sybilrank + rule)
    assert "the registered share, SybilRank admits " + listed([p["max_admitted"] for p in rule]) + "." in paper
    assert "seeds ($%.0f\\%%$ to $%.0f\\%%$ of the active accounts)" % (min(seed_share), max(seed_share)) in paper
    otc = networks["bitcoinotc"]
    assert "on Bitcoin-OTC ($%d$ against $%d$)" % (otc["sybilrank"]["probe"]["matched"]["max_admitted"],
                                                 otc["btp"]["constant"]) in paper
    assert [b["sybilrank"]["probe"]["matched"]["max_admitted"] > b["btp"]["constant"] for b in blocks] == [
        False, True, False]
    assert ("against SybilRank admits " + listed([p["max_admitted"] for p in sybilrank]) + " fakes, against "
            + listed([b["btp"]["best_star"] for b in blocks]) + " for BTP.") in paper
