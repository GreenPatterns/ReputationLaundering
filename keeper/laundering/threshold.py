"""Pre-registered decision threshold for the flip-rate metric (H1).

Sign convention, stated once because getting it wrong is silent: ``y == 1``
means DISTRUST and ``p`` is P(trust), so a model predicts distrust when
``p < tau``. Searching with ``(p >= tau)`` instead inverts every MCC value and
returns the degenerate ``tau == 0.0`` -- already observed in
``laundering/check_threshold.py``, which is why its ``below tau*`` counts were 0.

The threshold is computed ONCE per (dataset, seed) from the BASE model on the
held-out snapshot, before any deletion, so it cannot be tuned after seeing the
gains. p = 0.5 is always reported alongside for continuity.
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from sklearn.metrics import matthews_corrcoef  # noqa: E402

from keeper.laundering import core as C  # noqa: E402


def optimal_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """MCC-maximising split of P(trust); distrust is predicted when ``p < tau``.

    Ties in MCC are broken toward the threshold nearest 0.5, so the choice is
    deterministic and not an arbitrary grid endpoint.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return 0.5
    cands = np.unique(np.concatenate([np.round(p, 4), [0.5]]))
    scored = [(float(matthews_corrcoef(y, (p < t).astype(int))), float(t))
              for t in cands]
    best = max(s[0] for s in scored)
    tied = [t for m, t in scored if m >= best - 1e-12]
    return float(min(tied, key=lambda t: (abs(t - 0.5), t)))


def mcc_at(p: np.ndarray, y: np.ndarray, tau: float) -> float:
    return float(matthews_corrcoef(y, (np.asarray(p) < tau).astype(int)))


def held_out(G: C.Graph, D: C.Dataset) -> tuple:
    """Held-out snapshot edges/labels restricted to nodes seen in training."""
    sel = G.snap == D.tt
    e, y = G.edges[sel], D.y_neg()[sel]
    seen = np.zeros(G.n + 64, bool)
    seen[np.unique(G.edges[G.snap < D.tt].ravel())] = True
    k = seen[e[:, 0]] & seen[e[:, 1]]
    return e[k], y[k]


def pre_register(ds: str, seed: int = 0, epochs: int = 150,
                 n_probe: int = 256, force: bool = False) -> dict:
    """Train M0 once, derive tau*, and count the cohort's below-threshold targets. An existing
    threshold_<ds>_s<seed>.json is replaced only with force (--force) or KEEPER_OVERWRITE=1."""
    out_name = f"threshold_{ds}_s{seed}.json"
    C.result_path(out_name, force)            # refuse an existing result before the work, not after it
    C.set_deterministic(True)
    D = C.load(ds, verify=True)
    G = C.full_graph(D)
    Tr = C.train_split(G, D.tt)
    panel = C.Panel(Tr, n_probe=n_probe, seed=1234)

    m0 = C.train(C.make_model(D, Tr, seed, D.tt), Tr, D.tt, epochs=epochs)
    z0 = C.embed(m0, Tr, D.tt)

    e, y = held_out(G, D)
    p = C.p_trust(m0, z0, e[:, 0], e[:, 1]).cpu().numpy()
    tau = optimal_threshold(p, y)

    cohort, _, _ = C.pick_targets(Tr, max_neg=3, strict=True)
    r0 = panel.reputation(m0, z0, cohort)

    out = dict(
        ds=ds, seed=seed, epochs=epochs,
        tau_mcc=float(tau), tau_source="optimal_threshold(p,y), distrust if p<tau",
        mcc_at_tau=mcc_at(p, y, tau), mcc_at_half=mcc_at(p, y, 0.5),
        held_out_n=int(len(y)), distrust_rate=float(y.mean()),
        cohort_size=int(len(cohort)),
        n_below_tau=int((r0 < tau).sum()),
        frac_below_tau=float((r0 < tau).mean()),
        n_below_half=int((r0 < 0.5).sum()),
        frac_below_half=float((r0 < 0.5).mean()),
        rep_percentiles=[float(x) for x in np.percentile(r0, [5, 25, 50, 75, 95])],
        base_test=C.test_metrics(m0, z0, D, G),
    )
    C.dump(out, out_name, force=force)
    print(f"[{ds} s{seed}] tau*={tau:.4f} (MCC {out['mcc_at_tau']:.4f} vs "
          f"{out['mcc_at_half']:.4f} at 0.5)")
    print(f"    cohort {len(cohort)} | below tau* {out['n_below_tau']} "
          f"({100 * out['frac_below_tau']:.1f}%) | below 0.5 "
          f"{out['n_below_half']} ({100 * out['frac_below_half']:.1f}%)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-probe", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result file")
    a = ap.parse_args()
    pre_register(a.ds, a.seed, a.epochs, a.n_probe, force=a.force)