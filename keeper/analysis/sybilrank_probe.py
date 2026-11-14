"""SybilRank as the tier source in the admission probe: an exploratory, post hoc baseline (not registered).

Question: with the attack of the BTP admission probe (one honest account endorses one fake of a region of fakes),
how many fakes does SybilRank (Cao et al., NSDI 2012) admit where capped BTP admits at most floor(2 K) per attack
edge? No model is trained; CPU only. keeper.credibility.tiers is used unchanged.

SybilRank here: the endorsement graph of keeper.credibility.tiers.endorsement_edges without its direction (a
mutual pair is one edge), trust 1/|S0| on the seeds, K power iterations T(v) <- sum over neighbours u of
T(u) / deg(u), and the degree-normalised score T(v) / deg(v). The seeds S0 and K = ceil(log2 n_ref) are those of
BTP's frozen calibration, so both methods start from the same accounts and stop after the same number of steps.

Admission: an account is admitted when its score is at least theta and its positive ratings from accounts with
score >= theta outnumber its negative ratings from such accounts (the guard of btp_tiers, neg_from = "trusted").
theta is frozen on the honest graph, as BTP's calibration is, at the largest value whose admitted set keeps at
least ``target`` of the legacy Tier >= 1 accounts. Two targets: "matched" (the share capped BTP keeps on that
network) and "rule" (0.90, the preregistered keep rule of choose_eps).

Probe: one attack edge from an honest endorser into a star or a ring of s fakes (tiers_report.sybil_rows), for
the five accounts with the largest B(u) (the endorsers of BTP's probe) and the five with the largest SybilRank
inflow sum_{i<K} T_i(u) / (deg(u) + 1). The reported maximum is over these endorsers, the two topologies and the
probed sizes only, so it is a lower bound on the attacker's best response against SybilRank.

Because the graph is undirected, an edge that a fake creates by rating an honest account counts exactly as an
attack edge does (see test_endorsement_direction_and_mutual_pairs_give_one_undirected_edge); the probe does not
give the attacker such edges.

Run: python -m keeper.analysis.sybilrank_probe [--out results/analysis/sybilrank_probe.json] [--force]
"""
import argparse
import os
import time

import numpy as np

from keeper.analysis import tiers_report as TR
from keeper.analysis.laundering_extra import Inputs, sha256_file
from keeper.credibility import tiers
from keeper.laundering import core as C
from keeper.laundering import signed as S

NETWORKS = TR.NETWORKS
TOPOLOGIES = ("star", "ring")
PROBE_ENDORSERS = TR.PROBE_ENDORSERS                 # per ranking: BTP's B(u) and SybilRank's inflow
PROBE_SIZES = tuple(range(2, 61)) + (75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000)   # dense, then sparse
RULE_KEEP = tiers.EPS_KEEP
MODULE_FILE = "keeper/analysis/sybilrank_probe.py"

region_rows = TR.sybil_rows                          # the regions of the BTP probe, unchanged


# ------------------------------------------------------------------------ SybilRank
def _undirected(n_nodes: int, endorse) -> np.ndarray:
    """One row (a, b) with a < b per unordered pair of the endorsement rows."""
    endorse = np.asarray(endorse, dtype=np.int64).reshape(-1, 2)
    if len(endorse) and (endorse.min() < 0 or endorse.max() >= n_nodes):
        raise ValueError(f"endorsement endpoints must lie in [0, {n_nodes})")
    pairs = np.sort(endorse, axis=1)
    return np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0).reshape(-1, 2)


def degrees(n_nodes: int, endorse) -> np.ndarray:
    """Undirected degree of every account in the endorsement graph."""
    return np.bincount(_undirected(n_nodes, endorse).reshape(-1), minlength=n_nodes)


def sybilrank_trust(n_nodes: int, endorse, seeds, K: int, return_history: bool = False):
    """K power iterations of SybilRank from trust 1/|seeds| on each seed: T(v) <- sum_{u ~ v} T(u) / deg(u).

    An account without neighbours keeps its trust, so the total is conserved. Returns T^(K), or with
    return_history the iterates T^(0), ..., T^(K). Time O(K * M) for M undirected edges.
    """
    pairs = _undirected(n_nodes, endorse)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    deg = np.bincount(src, minlength=n_nodes)
    seeds = np.unique(np.asarray(seeds, dtype=np.int64).reshape(-1))
    if len(seeds) and (seeds[0] < 0 or seeds[-1] >= n_nodes):
        raise ValueError(f"seeds must lie in [0, {n_nodes})")
    if int(K) != K or K < 0:
        raise ValueError(f"K must be a non-negative integer, got {K}")
    trust = np.zeros(n_nodes, dtype=np.float64)
    if len(seeds):
        trust[seeds] = 1.0 / len(seeds)
    isolated = deg == 0
    history = [trust]
    for _ in range(int(K)):
        trust = np.where(isolated, trust, 0.0) + np.bincount(dst, weights=trust[src] / deg[src], minlength=n_nodes)
        history.append(trust)
    return history if return_history else trust


def normalised_score(trust: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """SybilRank's ranking score T(v) / deg(v); 0 for an account without neighbours."""
    trust, deg = np.asarray(trust, dtype=np.float64), np.asarray(deg)
    return np.divide(trust, deg, out=np.zeros_like(trust), where=deg > 0)


def _trust_and_score(Tr, n_nodes: int, seeds, K: int) -> tuple:
    endorse = tiers.endorsement_edges(Tr)
    trust = sybilrank_trust(n_nodes, endorse, seeds, K)
    return trust, normalised_score(trust, degrees(n_nodes, endorse))


def scores(Tr, n_nodes: int, seeds, K: int) -> np.ndarray:
    """The degree-normalised SybilRank score of every account of the graph Tr."""
    return _trust_and_score(Tr, n_nodes, seeds, K)[1]


def inflow_scores(Tr, n_nodes: int, seeds, K: int) -> np.ndarray:
    """sum_{i<K} T_i(u) / (deg(u) + 1): the trust one new edge at u would carry over K steps on the honest
    iterates (deg(u) + 1 counts the new edge). The SybilRank counterpart of tiers.endorser_inflow_scores."""
    endorse = tiers.endorsement_edges(Tr)
    history = sybilrank_trust(n_nodes, endorse, seeds, K, return_history=True)
    return np.sum(history[:-1], axis=0) / (degrees(n_nodes, endorse) + 1.0)


# ------------------------------------------------------------------------ admission and its threshold
def _signed(Tr, n_nodes: int) -> tuple:
    edges = np.asarray(Tr.edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError(f"edge endpoints must lie in [0, {n_nodes})")
    return edges, S.neg_mask(Tr.labels).reshape(-1)


def _admitted(edges: np.ndarray, is_neg: np.ndarray, n_nodes: int, score: np.ndarray, theta: float) -> np.ndarray:
    trusted = score >= theta
    from_trusted = trusted[edges[:, 0]]
    pos_in = np.bincount(edges[~is_neg & from_trusted, 1], minlength=n_nodes)
    neg_in = np.bincount(edges[is_neg & from_trusted, 1], minlength=n_nodes)
    return trusted & (pos_in > neg_in)


def admitted(Tr, n_nodes: int, score: np.ndarray, theta: float) -> np.ndarray:
    """Accounts with score >= theta whose positive ratings from accounts with score >= theta outnumber their
    negative ratings from such accounts (the eligibility guard of tiers.btp_tiers, neg_from = "trusted")."""
    return _admitted(*_signed(Tr, n_nodes), n_nodes, score, theta)


def calibrate_theta(Tr, n_nodes: int, score: np.ndarray, credible: np.ndarray, target: float) -> tuple:
    """(theta, kept): the largest theta whose admitted set keeps at least ``target`` of the ``credible`` accounts.

    Candidates are the positive scores of all accounts, because every account at or above theta votes in the
    guard. The guard makes the kept share non-monotone in theta, so the candidates are scanned from the largest
    down. If none qualifies, the smallest candidate is returned with its kept share below ``target``.
    """
    credible = np.asarray(credible, dtype=bool)
    n_credible = int(credible.sum())
    candidates = np.unique(score[score > 0])[::-1]
    if not n_credible or not len(candidates):
        raise ValueError("no credible account or no positive score, so theta is undefined")
    edges, is_neg = _signed(Tr, n_nodes)
    for theta in candidates:        # ponytail: O(candidates * E) scan; bisect if a graph ever makes this slow
        kept = int((_admitted(edges, is_neg, n_nodes, score, theta) & credible).sum()) / n_credible
        if kept >= target:
            break
    return float(theta), kept


# ------------------------------------------------------------------------ regions of fakes
def _region(Tr, n_nodes: int, rows, n_fakes: int, snap: int, seeds, K: int) -> tuple:
    """(attacked graph, its node count, trust, score) after the (u, v, sign) rows are injected in ``snap``."""
    G, n_all = TR.with_injected(Tr, rows, snap), n_nodes + n_fakes
    return (G, n_all) + _trust_and_score(G, n_all, seeds, K)


def region_case(Tr, n_nodes: int, rows, n_fakes: int, snap: int, seeds, K: int, theta: float) -> dict:
    """Fakes n_nodes .. n_nodes + n_fakes - 1 after ``rows`` are injected: how many are admitted at the frozen
    theta, and the trust the region holds after K steps."""
    G, n_all, trust, score = _region(Tr, n_nodes, rows, n_fakes, snap, seeds, K)
    return dict(admitted=int(admitted(G, n_all, score, theta)[n_nodes:].sum()),
                region_trust=float(trust[n_nodes:].sum()))


def endorser_probe(Tr, n_nodes: int, endorser: int, seeds, K: int, thetas: dict, sizes=PROBE_SIZES) -> dict:
    """One attack edge from ``endorser`` into a star and a ring of each size: per threshold name, the admitted
    fakes by topology and size and the best region (the smallest on ties, then the star). lower_bound is true when the best
    region was admitted whole and the next size was not probed."""
    snap, sizes = int(np.max(Tr.snap)), [int(s) for s in sizes]
    counts = {name: {topology: [] for topology in TOPOLOGIES} for name in thetas}
    for topology in TOPOLOGIES:
        for s in sizes:
            rows = region_rows([int(endorser)], np.arange(n_nodes, n_nodes + s), topology)
            G, n_all, _, score = _region(Tr, n_nodes, rows, s, snap, seeds, K)
            edges, is_neg = _signed(G, n_all)
            for name, theta in thetas.items():
                counts[name][topology].append(int(_admitted(edges, is_neg, n_all, score, theta)[n_nodes:].sum()))
    out = {}
    for name in thetas:
        best, topology, s = max(((a, t, s) for t in TOPOLOGIES for a, s in zip(counts[name][t], sizes)),
                                key=lambda row: (row[0], -row[2]))
        out[name] = dict(max_admitted=best, best_topology=topology, best_s=s,
                         lower_bound=bool(best == s and s + 1 not in sizes), admitted_by_size=counts[name])
    return dict(endorser=int(endorser), **out)


# ------------------------------------------------------------------------ networks
def network_block(Tr, n: int, sizes=PROBE_SIZES) -> dict:
    """Capped BTP's frozen calibration and probe (through the existing code), then SybilRank with the same seeds
    and K: its thresholds on the honest graph and its admission probe."""
    units = tiers.KEEPER_EDGE_CAP_UNITS
    eps, _ = tiers.choose_eps(Tr, n, tiers.EPS_GRID, tiers.EPS_KEEP, edge_cap_units=units,
                              fallback_grid=tiers.EPS_GRID_EXTENDED)
    cal = tiers.calibrate_btp(Tr, n, eps, edge_cap_units=units)
    legacy = tiers.legacy_bfs(Tr, n) > 0
    btp = tiers.btp_tiers(Tr, n, calibration=cal)[0] > 0
    btp_retention = int((btp & legacy).sum()) / int(legacy.sum())
    btp_probe = TR.star_size_probe(Tr, n, cal)

    score = scores(Tr, n, cal.seeds, cal.K)
    calibrations = {}
    for name, target in (("matched", btp_retention), ("rule", RULE_KEEP)):
        theta, kept = calibrate_theta(Tr, n, score, legacy, target)
        honest = admitted(Tr, n, score, theta)
        calibrations[name] = dict(target=target, theta=theta, retention=kept, met=bool(kept >= target),
                                  n_admitted=int(honest.sum()), n_scored=int((score >= theta).sum()))
    thetas = {name: c["theta"] for name, c in calibrations.items()}

    by_btp = [int(u) for u in btp_probe["endorsers"]]
    by_sybilrank = [int(u) for u in TR.top_accounts(inflow_scores(Tr, n, cal.seeds, cal.K), PROBE_ENDORSERS)]
    per_endorser = [endorser_probe(Tr, n, u, cal.seeds, cal.K, thetas, sizes)
                    for u in dict.fromkeys(by_btp + by_sybilrank)]
    probe = {}
    for name in thetas:
        best = max(per_endorser, key=lambda entry: entry[name]["max_admitted"])
        probe[name] = dict(max_admitted=best[name]["max_admitted"], argmax_endorser=best["endorser"],
                           argmax_topology=best[name]["best_topology"], argmax_s=best[name]["best_s"],
                           lower_bound=best[name]["lower_bound"])
    return dict(n_ref=cal.n_ref, K=cal.K, n_seeds=cal.n_seeds, n_legacy_tier1=int(legacy.sum()),
                btp=dict(eps=eps, theta0=cal.theta0, edge_cap_units=units, n_tier1=int(btp.sum()),
                         retention=btp_retention, constant=btp_probe["constant"],
                         best_star=btp_probe["max_admitted"], best_star_lower_bound=btp_probe["lower_bound"]),
                sybilrank=dict(calibrations=calibrations, probe=probe,
                               endorsers=dict(btp_inflow=by_btp, sybilrank_inflow=by_sybilrank),
                               per_endorser=per_endorser))


def analyse(datasets, inputs: Inputs, sizes=PROBE_SIZES) -> dict:
    networks = {}
    for ds in datasets:
        start = time.perf_counter()
        D = C.load(ds)
        for rel in [f"configs/{ds}.json"] + [D.cfg[k] for k in ("data_path", "edge_path") if D.cfg.get(k)]:
            inputs.path(rel)
        networks[ds] = network_block(C.train_split(C.full_graph(D), D.tt), D.n, sizes)
        block = networks[ds]
        print(f"[{ds}] BTP best star {block['btp']['best_star']} (bound {block['btp']['constant']}); SybilRank "
              + ", ".join(f"{name} {p['max_admitted']}{'+' if p['lower_bound'] else ''}"
                          for name, p in block["sybilrank"]["probe"].items())
              + f" ({time.perf_counter() - start:.0f} s)", flush=True)
    return dict(networks=networks)


SETTINGS = dict(status="exploratory, post hoc (2026-11-13); not part of the registered protocol",
                topologies=list(TOPOLOGIES), probe_endorsers_per_ranking=PROBE_ENDORSERS,
                probe_sizes=list(PROBE_SIZES), rule_keep=RULE_KEEP, t_seed=tiers.T_SEED,
                btp_edge_cap_units=tiers.KEEPER_EDGE_CAP_UNITS, guard="neg_from = trusted")


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="python -m keeper.analysis.sybilrank_probe", description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(NETWORKS), choices=NETWORKS)
    ap.add_argument("--out", default=os.path.join(TR.ANALYSIS, "sybilrank_probe.json"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    a = parse(argv)
    out_path = C.result_path(os.path.abspath(a.out), a.force)      # refuse before the work, not after it
    inputs = Inputs()
    result = analyse(a.datasets, inputs)
    result["settings"] = SETTINGS
    result["provenance"] = TR.provenance(inputs)
    result["provenance"]["module_sha256"][MODULE_FILE] = sha256_file(os.path.join(C.ROOT, MODULE_FILE))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    C.dump(result, out_path, force=a.force)
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
