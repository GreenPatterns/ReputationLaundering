"""Shared helpers of tests/test_scem_upgrades.py and tests/test_scem_witness_attack.py (Task 3.3). Not collected:
the name does not start with test_."""
import numpy as np

from keeper.analysis import defense_extra as DE
from keeper.credibility import tiers
from keeper.scem import scem as SC

PSI_NEG = DE.ENGINE["psi"]["negative"]          # a negative rating's channel routing in the engine
W0, W3 = tiers.TIER_WEIGHTS[0], tiers.TIER_WEIGHTS[3]
TOL = 1e-12


def engine(rows, n_nodes, K=21, **kw):
    """The streaming engine on (u, v, sign, snapshot) rows, with the per-record output."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    return SC.run_scem_streaming_weights_np(rows[:, :2], rows[:, 2], rows[:, 3], n_nodes, K_snap=K,
                                            return_records=True, **kw)


def closed_form(lag, kappas):
    """defense_extra's engine-rule revoked share of a negative rating whose sender is warned at `lag` by
    witnesses of reliabilities `kappas` (Task 2.2; matched to the streaming engine to 1e-12)."""
    return DE.revoked_by_routing(lag, DE.witness_hazard(kappas)["lam"], PSI_NEG)


# ------------------------------------------------------------------ the adversarial-witness stream


W, T = 0, 1

SYBILS = (2, 3, 4, 5, 6)

HISTORY = [(7, W, +1, 0), (8, W, +1, 0), (W, 9, +1, 1)]   # W was rated and has rated: an established account

N_NODES = 10


def sybil_stream(t_k=6, lag=1, ring=False):
    """W warns T at t_k (the victim record); five fresh Sybils bad-mouth W at t_k + lag. With `ring`, the
    Sybils first endorse each other in a ring at t_k - 1 (after t_seed), which gives them some reliability.
    T has no consensus and W's positive evidence has expired by t_k + 1, so the Sybils' evidence alone
    drives W's hazard, as the closed form assumes."""
    rows = HISTORY + [(W, T, -1, t_k)]
    victim = len(rows) - 1
    if ring:
        rows = rows + [(s, SYBILS[(i + 1) % len(SYBILS)], +1, t_k - 1) for i, s in enumerate(SYBILS)]
    return rows + [(s, W, -1, t_k + lag) for s in SYBILS], victim


def sybil_levels(sybil_level=0, other=2, K=21):
    levels = np.full((K, N_NODES), other, dtype=np.int64)
    levels[:, list(SYBILS)] = sybil_level
    return levels
