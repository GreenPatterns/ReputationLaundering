# Read-only: release script's ex-post rule (cp_vpe_scem.py:433-439) for every engine configuration
# used in Stage B. Engine = the user's unchanged run_scem_streaming_weights_np. Prints only.
"""Release-rule ablation of paper Table III, Block 1, with the one SCEM engine (keeper.scem.scem).

A rating counts as flagged when its sender received at least three distrust ratings in the training
stream, and more distrust than trust. Attenuation is the revoked share 1 - omega - pi of the flagged
ratings; allRet is the mean surviving weight over all training ratings.

Usage (CPU, about a minute):
    CUDA_VISIBLE_DEVICES="" python -m keeper.scem.release_rule
prints exactly results/scem/release_rule/release_rule_ablation_exact.txt.
"""
from typing import NamedTuple

import numpy as np

from keeper.laundering import core as C, signed as S
from keeper.scem import scem as SC

DATASETS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")
CFG = {"N1 (phi .25)": dict(N=1, phi=0.25), "N2 default": dict(N=2, phi=0.5), "N3 (phi .75)": dict(N=3, phi=0.75),
       "V2-only (cap_c=0)": dict(cap_c=0.0), "V1-only (c_self=0)": dict(c_self=0.0), "no O2": dict(supersession_impulse=0.0)}
DEFAULT = "N2 default"


class Stream(NamedTuple):
    edges: np.ndarray
    sign: np.ndarray
    snap: np.ndarray
    n: int
    tt: int
    flagged: np.ndarray       # ratings whose sender the ex-post rule flags


def load_stream(ds: str) -> Stream:
    D = C.load(ds); G = C.full_graph(D); Tr = C.train_split(G, D.tt)
    sign = 1 - 2 * S.class_ids(Tr.labels); n = int(G.n); e = Tr.edges
    pos_in = np.bincount(e[sign > 0, 1], minlength=n); neg_in = np.bincount(e[sign < 0, 1], minlength=n)
    m = np.isin(e[:, 0], np.where((neg_in >= 3) & (neg_in > pos_in))[0])
    return Stream(e, sign, Tr.snap, n, D.tt, m)


def effective_weight(st: Stream, kw: dict) -> np.ndarray:
    return SC.run_scem_streaming_weights_np(st.edges, st.sign, st.snap, st.n, K_snap=st.tt, **kw)["effective_weight"]


def attenuation(w: np.ndarray, flagged: np.ndarray):
    return 1 - w[flagged].mean()


def attenuation_default_rule(ds: str) -> float:
    """Attenuation of the default engine (N = 2, phi = 0.5) under the ex-post rule, as a fraction."""
    st = load_stream(ds)
    return float(attenuation(effective_weight(st, CFG[DEFAULT]), st.flagged))


def rows(ds: str) -> list:
    st = load_stream(ds)
    row = []
    for name, kw in CFG.items():
        w = effective_weight(st, kw)
        row.append(f"{name}: atten {100*attenuation(w, st.flagged):.3f}% allRet {w.mean():.5f}")
    return row


def main():
    for ds in DATASETS:
        print(ds)
        for r in rows(ds):
            print("   ", r)
    print("done")  # terminator of the stored file; the tag script did not print it, and removing it
                   # breaks the byte test in tests/test_release_rule.py


if __name__ == "__main__":
    main()
