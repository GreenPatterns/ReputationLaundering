"""Probe-panel members among the accounts that leave under top-R churn.

Read-only analysis behind one sentence of the supplement (Probe Panel).
The probe panel is drawn once from the pre-deletion training graph
(keeper/laundering/core.py, Panel, seed 1234) and is not redrawn after a deletion.
Incidental churn (keeper/laundering/incidental.py) removes the top-R accounts by
negative training ratings issued. A departing panel member stays in the panel,
so its rater term changes for every target. This script counts, per dataset,
how many of the top-R departing accounts are panel members.

It reproduces the panel and the top-R selection exactly as the two files do
and changes no file.

Run from the repository root:
    CUDA_VISIBLE_DEVICES="" ~/.conda/envs/ps/bin/python -m keeper.paperkit.panel_churn_overlap
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import keeper.laundering.core as C  # noqa: E402
import keeper.laundering.signed as S  # noqa: E402

DATASETS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa", "epinions-dated")
TOP_R = (1, 5, 20)


def overlap(ds: str) -> dict:
    D = C.load(ds)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    panel = set(C.Panel(Tr, n_probe=256, seed=1234).sources.tolist())
    issued = np.bincount(Tr.edges[S.neg_mask(Tr.labels), 0], minlength=Tr.n)
    order = np.argsort(-issued, kind="stable")
    return {r: len(panel & set(order[:r].tolist())) for r in TOP_R}


if __name__ == "__main__":
    for ds in DATASETS:
        counts = overlap(ds)
        print(ds + ": " + ", ".join(f"top-{r}: {c} panel members" for r, c in counts.items()))
