"""Helpers shared by the credibility tests (test_credibility_legacy, test_credibility_btp, test_credibility_guard,
test_tiers_report); not collected."""
from types import SimpleNamespace

import numpy as np

from keeper.credibility import tiers
from keeper.laundering import signed as S

NETWORKS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")
HONEST_N = 60       # accounts 0..59 of the random toy graphs; fakes are numbered from 60
SEED_CORE = 10      # only accounts 0..9 rate or are rated in snapshots 0-4, so the seeds lie among them
SYBIL_SNAP = 19     # Sybil and attack edges come after the seed window (snapshots 0-4)
TOY_EPS = 0.2


def graph(rows):
    """A training graph from (u, v, sign, snapshot) rows, in the given (stored) order."""
    rows = np.asarray(rows, dtype=np.int64)
    return SimpleNamespace(edges=rows[:, :2], labels=S.one_hot_from_sign(rows[:, 2]), snap=rows[:, 3])


# Node 0 is the only seed (two positive in-edges in snapshots 0-1). Its one positive rating of node 4
# admits 4, and the ring 4 -> 5 -> 6 -> 4 follows in the same pass. Node 7 is outside the endorsed set, so
# its positive rating of node 3 does not count when seeded.
RING = [(1, 0, +1, 0), (2, 0, +1, 1), (0, 3, -1, 2),
        (0, 4, +1, 5), (4, 5, +1, 6), (5, 6, +1, 6), (6, 4, +1, 6), (7, 3, +1, 7), (3, 7, +1, 8)]


def load_training_graphs() -> dict:
    """ds -> (training graph, node count), through the dataset loader and the dataset's tt."""
    from keeper.laundering import core as C
    out = {}
    for ds in NETWORKS:
        D = C.load(ds)
        out[ds] = (C.train_split(C.full_graph(D), D.tt), D.n)
    return out


def in_time_order(rows):
    """Rows stably sorted by snapshot, as the dataset loader stores edges."""
    return sorted(rows, key=lambda r: r[3])


def with_rows(Tr, rows, snap):
    """Tr's edges followed by the (u, v, sign) rows, in their order, all in snapshot ``snap``."""
    r = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    return SimpleNamespace(edges=np.concatenate([np.asarray(Tr.edges, dtype=np.int64).reshape(-1, 2), r[:, :2]]),
                           labels=np.concatenate([np.asarray(Tr.labels, dtype=np.float32),
                                                  S.one_hot_from_sign(r[:, 2])]),
                           snap=np.concatenate([np.asarray(Tr.snap, dtype=np.int64), np.full(len(r), snap)]))


def honest_rows(seed, n_h=HONEST_N, core=SEED_CORE, m_early=40, m_late=360, p_neg=0.15):
    """(u, v, sign, snapshot) rows of a random signed graph, in snapshot order, without self-ratings.

    Snapshots 0-4 hold m_early ratings among the core accounts 0..core-1, so the seeds lie among them;
    snapshots 5-19 hold m_late ratings among all n_h accounts.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for count, accounts, first, last in ((m_early, core, 0, 5), (m_late, n_h, 5, 20)):
        u, v = rng.integers(0, accounts, count), rng.integers(0, accounts, count)
        keep = u != v
        sign = np.where(rng.random(int(keep.sum())) < p_neg, -1, 1)
        snap = np.sort(rng.integers(first, last, int(keep.sum())))
        rows += [tuple(int(x) for x in r) for r in zip(u[keep], v[keep], sign, snap)]
    return rows


def hub_rows():
    """Hubs 0-4 with 1, 2, 4, 8 and 16 leaves (accounts 5-35).

    Each leaf rates its hub in snapshot 0, so the hubs, and only they, are seeds, and the hub rates the leaf
    back in snapshot 5. BTP is stationary after one step: a hub holds 1/10 and each leaf of a hub with m
    leaves 1/(10 m). The legacy tiers endorse all 36 accounts.
    """
    rows, leaf = [], 5
    for hub, m in enumerate((1, 2, 4, 8, 16)):
        for _ in range(m):
            rows += [(leaf, hub, +1, 0), (hub, leaf, +1, 5)]
            leaf += 1
    return in_time_order(rows)


def attacked_region(rows, attackers, s, topology="ring", attack_snap=SYBIL_SNAP):
    """rows, then the attack edges, then the region, in snapshot order. Ring: attacker a endorses fake a mod s,
    and fake_0 -> fake_1 -> ... -> fake_0. Star: every attacker endorses the hub fake_0, which endorses every
    other fake. The region endorses no honest account. ``attackers`` is one account or a list of them."""
    attackers = [attackers] if isinstance(attackers, int) else [int(u) for u in attackers]
    fakes = list(range(HONEST_N, HONEST_N + s))
    targets = [fakes[a % s] if topology == "ring" else fakes[0] for a in range(len(attackers))]
    region = [(u, f, +1, attack_snap) for u, f in zip(attackers, targets)]
    if topology == "ring":
        region += [(fakes[j], fakes[(j + 1) % s], +1, SYBIL_SNAP) for j in range(s)]
    else:
        region += [(fakes[0], fakes[j], +1, SYBIL_SNAP) for j in range(1, s)]
    attack = np.array([[u, f] for u, f in zip(attackers, targets)])
    return graph(in_time_order(rows + region)), HONEST_N + s, fakes, attack


def seed_attacker(rows):
    """The first seed with an endorsee: a member of the legacy endorsed set D from the start."""
    return seed_endorsers(rows, 1)[0]


def seed_endorsers(rows, k):
    """The first k seeds with an endorsee."""
    pre = graph(rows)
    out_deg = np.bincount(tiers.endorsement_edges(pre)[:, 0], minlength=HONEST_N)
    return [int(u) for u in tiers.seed_set(pre, HONEST_N) if out_deg[u] > 0][:k]


def max_inflow_endorsers(rows, k, edge_cap_units=None):
    """The k honest accounts with the largest B(u), the trust one new endorsement would carry over K steps
    (keeper.credibility.tiers.endorser_inflow_scores, ruling R33b), under the toy's calibration at TOY_EPS with
    the given cap; ties go to the smaller id."""
    pre = graph(rows)
    scores = tiers.endorser_inflow_scores(pre, HONEST_N, tiers.calibrate_btp(pre, HONEST_N, TOY_EPS,
                                                                             edge_cap_units=edge_cap_units))
    return [int(u) for u in np.argsort(-scores, kind="stable")[:k]]


def max_final_trust_endorsers(rows, k):
    """The k honest accounts with the largest final uncapped trust per out-edge, t^(K)(u) / (dout(u) + 1): the
    ranking of ruling R33, which R33b replaced. Kept to show that B(u) finds stronger endorsers."""
    pre = graph(rows)
    trust = tiers.btp_tiers(pre, HONEST_N, TOY_EPS)[1]
    out_deg = np.bincount(tiers.endorsement_edges(pre)[:, 0], minlength=HONEST_N)
    return [int(u) for u in np.argsort(-trust / (out_deg + 1.0), kind="stable")[:k]]
