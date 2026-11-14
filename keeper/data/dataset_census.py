#!/usr/bin/env python3
"""Census of the reputation-laundering attack surface, per dataset.

Reproduces the proposal's section 2 claims on the real graph and reports the
numbers the probe needs: how many targets rest on <=3 distrust in-edges, how
concentrated distrust is among raters, and how the indirect channel (C3) is
sourced. Run for every dataset before quoting a number about it.

The honest cohort is ``distrust-majority AND npos>=1``: the proposal's attack
keeps the target's identity AND its positive history, so a target with no trust
in-edges is not in the threat model (see FINDINGS_2026-09-30.md).

Usage:
  env -u PYTHONPATH -u VIRTUAL_ENV ~/.conda/envs/ps/bin/python \\
      laundering/dataset_census.py bitcoinalpha bitcoinotc epinions-dated
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402


def census(ds: str) -> dict:
    D = C.load(ds, verify=True)
    G = C.full_graph(D)
    Tr = C.train_split(G, D.tt)

    neg = collections.defaultdict(list)     # target -> distrust in-edge row ids
    pos = collections.defaultdict(list)     # target -> trust in-edge row ids
    issued = collections.Counter()          # RATER -> distrust edges it issued
    is_neg = S.neg_mask(Tr.labels)
    for i, (a, b) in enumerate(Tr.edges):
        (neg if is_neg[i] else pos)[int(b)].append(i)
        if is_neg[i]:
            issued[int(a)] += 1

    with_neg = [v for v in neg if neg[v]]
    majority = [v for v in with_neg if len(neg[v]) > len(pos.get(v, []))]
    le3 = [v for v in majority if len(neg[v]) <= 3]
    honest = [v for v in le3 if len(pos.get(v, [])) >= 1]

    n_neg_total = sum(len(v) for v in neg.values())
    # §2's concentration statistic is about RATERS (who issues distrust). The
    # earlier version ranked the 20 most-distrusted TARGETS (keyed on the edge
    # target), which is a different quantity (audit A2-01).
    top20_raters = [u for u, _ in issued.most_common(20)]
    top20_targets = sorted(neg, key=lambda u: -len(neg[u]))[:20]

    # distrusted targets by how thin their positive history is
    npos_hist = collections.Counter(min(len(pos.get(v, [])), 5) for v in honest)

    # indirect-channel supply: distrust in-edges of the cohort's neighbours
    nbrs = collections.defaultdict(set)
    for v in honest:
        for i in neg[v]:
            nbrs[v].add(int(Tr.edges[i, 0]))
    c3 = []
    for v in honest:
        cand = {i for u in nbrs[v] for i in neg.get(u, [])
                if int(Tr.edges[i, 1]) != v}
        c3.append(len(cand))

    return dict(
        ds=ds, edges=len(D.edges), nodes=D.n, T=D.T, tt=D.tt,
        train_window_edges=len(Tr.edges),
        distrust_share=round(n_neg_total / max(len(Tr.edges), 1), 4),
        nodes_with_neg_in=len(with_neg),
        distrust_majority=len(majority),
        majority_le3=len(le3),
        honest_cohort=len(honest),
        distrust_edges=n_neg_total,
        top20_raters_distrust_share=round(
            sum(issued[u] for u in top20_raters) / max(n_neg_total, 1), 3),
        top20_targets_distrust_share=round(
            sum(len(neg[u]) for u in top20_targets) / max(n_neg_total, 1), 3),
        honest_npos_hist={str(k): v for k, v in sorted(npos_hist.items())},
        c3_candidates_median=int(sorted(c3)[len(c3) // 2]) if c3 else 0,
        c3_zero_candidate_targets=int(sum(1 for x in c3 if x == 0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--json", help="also write the census here")
    a = ap.parse_args()
    out = [census(ds) for ds in a.datasets]
    for r in out:
        print(f"\n=== {r['ds']} ===")
        for k, v in r.items():
            if k != "ds":
                print(f"  {k:26s} {v}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())