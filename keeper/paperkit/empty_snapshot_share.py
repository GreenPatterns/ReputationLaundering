"""Share of cohort targets and snapshots in which the target has no in-edge.

Read-only analysis behind one sentence of Section 6.4 (CPPR whitewash
residual). The inject-then-erase planner (keeper/cppr/arm_graphs.py,
build_inject_then_erase_graph, re-exported by keeper/cppr/cp_vpe_gpu.py)
places the three fake +1 ratings in a random training snapshot drawn from
[FIRST_FAKE_SNAPSHOT, tt) = [5, tt). The credibility gate normalizes tier
weights over a target's in-edges within one snapshot, so the tier weight of
the retained fake ratings cancels whenever they are the target's only
in-edges in that snapshot. This script reports, for each dataset, the mean
over cohort targets of the share of snapshots 5..tt-1 in which the target has
no in-edge in the training graph.

It imports the project's data loader and changes no file.

Run from the repository root:
    CUDA_VISIBLE_DEVICES="" ~/.conda/envs/ps/bin/python -m keeper.paperkit.empty_snapshot_share
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import keeper.laundering.core as C  # noqa: E402
from keeper.cppr.protocol import FIRST_FAKE_SNAPSHOT  # noqa: E402  (the builder's lowest snap_f)

DATASETS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")


def empty_share(ds: str) -> tuple:
    D = C.load(ds)
    G = C.full_graph(D)
    tt = D.T - 1
    Tr = C.train_split(G, tt)
    cohort, _, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    snaps = np.arange(FIRST_FAKE_SNAPSHOT, tt)
    shares = []
    for v in cohort:
        occupied = set(Tr.snap[Tr.edges[:, 1] == v].tolist())
        shares.append(np.mean([s not in occupied for s in snaps]))
    return len(cohort), float(np.mean(shares))


if __name__ == "__main__":
    for ds in DATASETS:
        n, share = empty_share(ds)
        print(f"{ds}: cohort {n}, share of snapshots {FIRST_FAKE_SNAPSHOT}..20 "
              f"with no in-edge: {100 * share:.1f}%")
