"""The graphs that the arms of the CPPR driver keeper.cppr.cp_vpe_gpu train on and are scored on.

build_inject_then_erase_graph gives the whitewash and bad-mouth arms' injected graphs, build_control_graph the
control arm's, plain_erasure_graph the Plain Erasure graph of the direct arm, and evaluation_graphs the graph each
arm's models are scored on. Task 4.1b moved them here from the driver unchanged, except that the lowest snapshot a
group can draw is now the named constant keeper.cppr.protocol.FIRST_FAKE_SNAPSHOT (5, as before), which the
driver's pre-flight check reads. The driver re-exports all four, so keeper.cppr.cp_vpe_gpu.<name> still resolves.
"""
from dataclasses import replace

import numpy as np

from keeper.cppr import vpe as V
from keeper.cppr.protocol import FIRST_FAKE_SNAPSHOT, SYBIL_TOPOLOGIES, honest_rows_lead
from keeper.credibility.tiers import legacy_levels as compute_credibility_levels
from keeper.laundering import core as C
from keeper.laundering import signed as S


def build_inject_then_erase_graph(D: "C.Dataset", G: "C.Graph", tt: int,
                                  cred_levels: np.ndarray, targets: list,
                                  sign_val: int, k_fake: int,
                                  honest_reqs: list, rng: np.random.Generator,
                                  sybil_ring: bool = True, seeded: bool = True,
                                  endorsers=None, topology: str = "ring", tier_fn=None):
    """Inject k_fake fakes per target that rate it ``sign_val`` at snap_f ~ U[5, tt), and append their account
    deletions to the background requests. Returns (D_inj, G_inj, cred_inj, requests).

    A group's edges, all at its snap_f, in this order:
    * the attack edges, +1 from each endorser to a fake, round-robin over the fakes (ring) or to the hub, the
      group's first fake (star). ``endorsers`` holds one list of honest accounts per target, or is a callable
      (group, snap_f) -> list called once the group's snap_f is drawn (an EndorserPicker, ruling R39); None: none;
    * per fake, its rating of the target and, when sybil_ring and sign_val > 0, its region edge: fake_j ->
      fake_(j+1 mod k) (ring), or hub -> fake_j for j >= 1 (star).
    The stable sort by snapshot keeps that order after the original edges of snap_f, so legacy's single pass reads
    an attack edge before the region's edges. cred_inj is ``tier_fn`` on the injected training graph (None: the
    legacy tiers with ``seeded``). Without endorsers and with the ring, the graph, the tiers and the random draws
    are those of the stored runs.
    """
    if topology not in SYBIL_TOPOLOGIES:
        raise ValueError(f"topology must be one of {SYBIL_TOPOLOGIES}, got {topology!r}")
    if endorsers is not None and not callable(endorsers) and len(endorsers) != len(targets):
        raise ValueError(f"{len(endorsers)} endorser lists for {len(targets)} targets")
    n_fake = k_fake * len(targets)
    fake_ids = np.arange(D.n, D.n + n_fake, dtype=np.int64)
    new_e, new_lab, new_snap, new_time = [], [], [], []
    fake_reqs = list(honest_reqs)
    ptr = 0
    for i, v in enumerate(targets):
        grp = [int(fake_ids[ptr + j]) for j in range(k_fake)]
        snap_f = int(rng.integers(FIRST_FAKE_SNAPSHOT, tt))
        t_f = int(G.time[G.snap == snap_f].max()) if (G.snap == snap_f).any() else 0
        group = () if endorsers is None else endorsers(i, snap_f) if callable(endorsers) else endorsers[i]
        for a, u in enumerate(group):
            new_e.append((int(u), grp[0] if topology == "star" else grp[a % k_fake]))
            new_lab.append(+1)
            new_snap.append(snap_f)
            new_time.append(t_f)
        for j in range(k_fake):
            f_u = grp[j]
            new_e.append((f_u, int(v)))
            new_lab.append(sign_val)
            new_snap.append(snap_f)
            new_time.append(t_f)
            if sybil_ring and sign_val > 0 and (topology == "ring" or j > 0):
                new_e.append((f_u, grp[(j + 1) % k_fake]) if topology == "ring" else (grp[0], f_u))
                new_lab.append(+1)
                new_snap.append(snap_f)
                new_time.append(t_f)
            fake_reqs.append(V.Request(account=f_u, tag="inject_then_erase", target=int(v)))
        ptr += k_fake

    e_arr = np.array(new_e, dtype=np.int64)
    lab_arr = S.one_hot_from_sign(np.array(new_lab))
    rid_arr = np.arange(len(D.edges), len(D.edges) + len(e_arr), dtype=np.int64)
    G_inj = C.Graph(
        np.concatenate([G.edges, e_arr]),
        np.concatenate([G.labels, lab_arr]),
        np.concatenate([G.time, np.array(new_time, dtype=np.int64)]),
        np.concatenate([G.snap, np.array(new_snap, dtype=np.int64)]),
        D.n + n_fake,
        np.concatenate([G.rid, rid_arr]),
    )
    order = np.argsort(G_inj.snap, kind="mergesort")
    G_inj = G_inj.subset(order)
    if endorsers is not None and not honest_rows_lead(order, G_inj.snap, len(G.edges)):
        # the premises of R39's condition join <= snap_f (EndorserPicker.candidates) fail
        raise ValueError("G's rows must be in snapshot order: honest rows keep their order, before injected ones")
    D_inj = C.Dataset(D.name, G_inj.edges, G_inj.labels, G_inj.time, G_inj.n, D.T, D.cfg,
                      meta=dict(D.meta, injected_fakes=n_fake))
    Tr_inj = C.train_split(G_inj, tt)
    cred_inj = (compute_credibility_levels(Tr_inj, D.n + n_fake, seeded=seeded) if tier_fn is None
                else tier_fn(Tr_inj, D.n + n_fake))
    # Edge-level requests hold row positions of the original training list, and
    # inserting the fakes shifts every later position. Re-point each request to
    # the same edges (same row ids) in the injected training list.
    Tr0 = C.train_split(G, tt)
    pos_of_rid = {int(r): i for i, r in enumerate(Tr_inj.rid)}
    fake_reqs = [
        replace(q, rows=np.array([pos_of_rid[int(r)] for r in Tr0.rid[np.asarray(q.rows)]],
                                 dtype=np.int64))
        if q.rows is not None and len(q.rows) else q
        for q in fake_reqs
    ]
    return D_inj, G_inj, cred_inj, fake_reqs


def build_control_graph(D: "C.Dataset", G: "C.Graph", cred_levels: np.ndarray, honest_reqs: list) -> tuple:
    """The control arm (--control-arm): the whitewash pipeline with no fake. Graph and tiers are the original
    ones and the requests are the background requests alone, so the arm's gain on the whitewash targets is the
    retraining noise plus the background erasures. Returns (D, G, cred, requests) like
    build_inject_then_erase_graph, and draws no random number."""
    return D, G, cred_levels, list(honest_reqs)


def plain_erasure_graph(D: "C.Dataset", G: "C.Graph", tt: int, erase_rids, hold_test_snapshot: bool = False):
    """(Dp, Gp, Trp) of the Plain Erasure arm.

    Default (the stored runs): exact retraining on keeper.laundering.core.erased_dataset, whose snapshots are re-cut
    on the erased data; the test snapshot then shrinks, and some original test edges move into training.
    hold_test_snapshot: the erased rows leave and every other edge keeps its snapshot, as in Plain VPE and CPPR,
    so the test snapshot stays the original one and no test edge enters training.
    """
    Dp, ridp = C.erased_dataset(D, erase_rids)
    if not hold_test_snapshot:
        Gp = C.full_graph(Dp, rid=ridp)
    else:
        Gp = G.subset(~G.mask_of(erase_rids))
        if not np.array_equal(Gp.rid, ridp):
            raise ValueError("G must be the full graph of D (row ids 0 .. E - 1) to hold its test snapshot")
    return Dp, Gp, C.train_split(Gp, tt)


def evaluation_graphs(hold_test_snapshot: bool, D: "C.Dataset", G: "C.Graph", D_arm, G_arm) -> tuple:
    """(D, G) to score an arm's model on: with --hold-test-snapshot the original graph, so every arm has the
    original test rows and Task 1 and Task 2 masks; else the arm's own graph, as in the stored runs."""
    return (D, G) if hold_test_snapshot else (D_arm, G_arm)
