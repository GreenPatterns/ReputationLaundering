"""E1: the SCEM streaming engine on the rating streams (Task 4.4; docs/preregistration.md Section 2.3, H4 and H5).

  python -m keeper.scem.e1_runner STUDY --ds DS --out ABS.json [--shard I/N] [--prereg PATH] [--force]

CPU only and deterministic. The engine, keeper.scem.scem.run_scem_streaming_weights_np with its defaults (L_f = 1),
has no learned parameter, and the only random draw is E1b's endorsers. The runner refuses to start when a GPU is
visible; run it with CUDA_VISIBLE_DEVICES="". It writes one JSON per network and study, or per shard of a study:
--shard I/N takes the units I, I + N, I + 2N, ... of the study's unit list, and keeper.analysis.e1_analysis merges the
shards. Every output records the stream, the engine's defaults, each release policy with its closure age A, the BTP
calibrations (eps, edge_cap_units, neg_from, t_seed, K(t), n_ref(t), theta0(t), and K and n_ref on the whole clean
stream), the protocol's sha256 and, in its provenance, the git commit. Writes go through keeper.laundering.core.dump
(atomic; an existing file is replaced only with --force or KEEPER_OVERWRITE=1).

Studies (Section 2.3):
  e1a     release policies without attack on the training stream (snapshots 0-20): the seven POLICIES, each with M-B off
          (the stored witness rule), capped M-B and uncapped M-B (the sensitivity); escrow_metrics.
  e1b     adversarial witnesses, Erlang N = 2, M-B off and capped M-B on, over E1B_CELLS. Unit: (cell, batch). Per
          victim record: the revoked share 1 - omega - pi at the end of the stream under attack, on the comparator
          (comparator_stream, with the attacked stream's witness levels) and on the clean stream, and
          keeper.scem.property1.property1_bound. Per batch: the honest accounts whose capped tier changed.
  e1c     the window property (H5): the E1C_POLICIES, M-B off, k = 5, g = 0, ring off, lags E1B_LAGS. Unit: (lag,
          batch).
          The analysis applies H5's eligibility to the recorded t_k, lag and A.
  e1d     granularity (M-C): the full stream re-cut into 22, 44 and 88 snapshots; Erlang N = 2 and arrival, M-B off and
          capped on; escrow_metrics, days per snapshot and each policy's horizon A in days.
  e1d_h5  e1c on the 88-snapshot stream, with the lags 0, 4, 12, 28 and 56.
  e1c_single, e1d_h5_single
          H5's test: e1c and e1d_h5 with one victim record per attacked stream. Unit: (lag, record), run only when
          the record is eligible under at least one committing policy (eligible()). In a batch of several records
          the other records' attacks reach a record while it is still in escrow (indirect paths: reliability kappa
          and the self-surprise term), so the batched studies do not isolate a record's own late attack.

Clarifications fixed in code before the first run (protocol Section 5); each is the reading closest to the text:
  1. Batches. A cell's victim records are those it does not skip (ruling R34, as the injector applies it), taken in
     record-id order and placed first-fit: a record joins the first batch that holds no record of its warner and fewer
     than floor(n_ref / (50 k)) records, the largest count with k x records <= 2% of n_ref. This is one record per
     batch at k = 50 on Bitcoin-Alpha, two on Bitcoin-OTC and four on Wiki-RfA; the protocol's parenthesis ("at
     k = 50, one record per batch") holds on Alpha only, and the rule it illustrates is followed.
  2. Endorsers. "cell index" is the position in E1B_CELLS, which does not distinguish M-B off from on, so both face
     the same attacked stream (the common-draw principle of Section 2.2).
  3. H4 names k = 50 with the ring off, a cell the E1b grid lacks; E1B_CELLS adds (k = 50, g = 0, lag 1, ring off).
  4. Honest tier changes: accounts of the clean stream whose capped causal tier differs, in any row, between the
     attacked and the clean stream; counted per batch.
  5. Honest mass revoked: the mean over the ratings of senders none of whose ratings is flagged (over the whole
     stream); the mean of the per-sender means is reported as well.
  6. Latency at age a: the mean released share omega / (omega + pi) at the end of snapshot t_k + a - 1, over the
     records with t_k + a - 1 <= K - 1 and positive surviving mass.
  7. E1d re-cuts the full stream (every rating, the held-out snapshot included) with keeper.laundering.core.full_graph
     at T = 22, 44, 88 and runs all T snapshots (K_snap = T). t_seed (4, 9, 19) scales M-B's rule, the calibration and
     the attack timing; the engine's stored rule (M-B off) keeps its bootstrap floor at t <= 4 in every cut, an
     engine constant this task does not change. Days per snapshot = the data's time span / T.
  8. E1c and H5 see attacked records only: a record that R34 skips at a lag (t_k + lag <= t_seed + 1, or
     t_k + lag >= K) carries no attack, so it is counted as skipped, never as eligible.
"""
import argparse
import dataclasses
import os
import time
from typing import NamedTuple

import numpy as np

from keeper.analysis.provenance import git_state
from keeper.cluster import protocol_settings as ps
from keeper.cppr.protocol import preregistration
from keeper.credibility import tiers
from keeper.laundering import core as C
from keeper.laundering import signed as S
from keeper.scem import property1 as P1
from keeper.scem import scem as SC
from keeper.scem import stage_b
from keeper.scem import witness_attack as WA
from keeper.scem import witness_cred as WC
from keeper.scem.policies import release_policy

STUDIES = ("e1a", "e1b", "e1c", "e1d", "e1d_h5", "e1c_single", "e1d_h5_single")
SHARDED = ("e1b", "e1c", "e1d_h5", "e1c_single", "e1d_h5_single")
ENTROPY = 20261027                     # E1b endorsers: default_rng((ENTROPY, network index, batch, cell index))
BATCH_SHARE_DENOM = 50                 # k x (records in a batch) <= 2% of n_ref = n_ref / 50
POLICIES = {                           # name -> engine keyword arguments (Section 2.3, E1a)
    "arrival": dict(policy="arrival"),
    "exponential": dict(policy="erlang", N=1, phi=0.25),
    "erlang2": dict(policy="erlang", N=2, phi=0.5),
    "erlang3": dict(policy="erlang", N=3, phi=0.75),
    "erlang5": dict(policy="erlang", N=5, phi=1.25),
    "quarantine": dict(policy="quarantine", quarantine_D=4),
    "no_commit": dict(policy="no_commit"),
}
COMMITTING = ("exponential", "erlang2", "erlang3", "erlang5", "quarantine")
E1B_POLICY = "erlang2"
E1B_K, E1B_G, E1B_LAGS = (1, 5, 10, 50), (0, 1, 2), (0, 1, 3, 7, 14)
E1C_POLICIES = COMMITTING + ("no_commit",)
E1C_K, E1C_G = 5, 0
CUT_T_SEED = {22: 4, 44: 9, 88: 19}    # E1d: snapshots -> t_seed
H5_CUT, H5_LAG_SCALE = 88, 4
E1D_POLICIES = ("erlang2", "arrival")
CAP_UNITS = {"capped": ps.EDGE_CAP_UNITS, "uncapped": None}


class Cell(NamedTuple):
    """An E1b attack cell: k Sybils per record, g attack edges, the lag, the ring, and the kappa-farming ring."""
    k: int
    g: int
    lag: int
    ring: bool
    farming: bool = False


E1B_CELLS = (tuple(Cell(k, g, lag, True) for k in E1B_K for g in E1B_G for lag in E1B_LAGS)
             + (Cell(5, 0, 1, False), Cell(5, 1, 1, False), Cell(50, 0, 1, False))     # the last: clarification 3
             + (Cell(5, 0, 1, True, True), Cell(50, 0, 1, True, True)))


@dataclasses.dataclass(frozen=True)
class Stream:
    """A clean rating stream with its settings: K snapshots, t_seed, and the eps of each M-B variant."""
    ds: str
    net_index: int
    edges: np.ndarray
    sign: np.ndarray
    snap: np.ndarray
    n: int
    K: int
    t_seed: int
    eps: dict
    time: np.ndarray | None = None

    @property
    def clean(self) -> tuple:
        return self.edges, self.sign, self.snap

    @property
    def n_ref(self) -> int:
        """Active accounts of the stream (Section 2.2's n_ref on the training stream)."""
        active = np.zeros(self.n, dtype=bool)
        active[self.edges.reshape(-1)] = True
        return int(active.sum())

    def record(self) -> dict:
        return dict(K=self.K, t_seed=self.t_seed, n_nodes=self.n, n_ratings=len(self.sign), n_ref=self.n_ref)


def _stream(ds, edges, sign, snap, n, K, t_seed, times) -> Stream:
    return Stream(ds, ps.NETWORKS.index(ds), np.asarray(edges, dtype=np.int64), np.asarray(sign, dtype=np.int64),
                  np.asarray(snap, dtype=np.int64), int(n), int(K), int(t_seed),
                  {"capped": ps.EPS_CAPPED[ds], "uncapped": ps.EPS_UNCAPPED[ds]}, times)


def training_stream(ds: str):
    """(D, the training stream): snapshots 0..20 of the loader's split, read as keeper.scem.stage_b reads it."""
    D, G, Tr, sign = stage_b.load_stream(ds)
    return D, _stream(ds, Tr.edges, sign, Tr.snap, max(G.n, Tr.edges.max() + 1), Tr.snap.max() + 1, tiers.T_SEED,
                      Tr.time)


def cut_stream(D, cut: int) -> Stream:
    """The full stream re-cut by the loader's own rule into `cut` snapshots, all of them kept (clarification 7)."""
    G = C.full_graph(dataclasses.replace(D, T=cut))
    if int(G.snap.max()) + 1 != cut:
        raise ValueError(f"{D.name}: the {cut}-snapshot cut leaves its last snapshot empty")
    return _stream(D.name, G.edges, 1 - 2 * S.class_ids(G.labels), G.snap, max(G.n, G.edges.max() + 1), cut,
                   CUT_T_SEED[cut], G.time)


# ------------------------------------------------------------------ engine, levels, settings


def run_engine(stream: Stream, edges, sign, snap, n_nodes: int, levels, policy: str) -> dict:
    """The engine over the stream's K snapshots with per-record output; levels None is M-B off."""
    return SC.run_scem_streaming_weights_np(edges, sign, snap, n_nodes, K_snap=stream.K, t_seed=stream.t_seed,
                                            witness_cred=levels, return_records=True, **POLICIES[policy])


def revoked_share(out: dict) -> np.ndarray:
    """1 - omega - pi of every record at the end of the stream (Section 2.3)."""
    rec = out["records"]
    return 1.0 - rec["omega"] - rec["pi"]


def witness_levels(stream: Stream, edges, sign, snap, n_nodes: int, variant: str = "capped") -> np.ndarray:
    """Causal BTP witness tiers of a stream under the calibration frozen on the clean stream (ruling R32)."""
    return WC.witness_levels(edges, sign, snap, n_nodes, stream.eps[variant], clean=stream.clean,
                             edge_cap_units=CAP_UNITS[variant], T=stream.K, t_seed=stream.t_seed)


def calibration_record(stream: Stream, variant: str) -> dict:
    """The frozen causal calibration of an M-B variant, and K and n_ref of the whole clean stream."""
    eps, cap = stream.eps[variant], CAP_UNITS[variant]
    cal = WC.clean_calibration(*stream.clean, stream.n, eps, edge_cap_units=cap, T=stream.K, t_seed=stream.t_seed)
    full = tiers.calibrate_btp(WC.stream_graph(*stream.clean), stream.n, eps, t_seed=stream.t_seed,
                               edge_cap_units=cap)
    return dict(variant=variant, eps=eps, edge_cap_units=cap, neg_from=tiers.KEEPER_NEG_FROM, t_seed=stream.t_seed,
                K_t=list(cal.K), n_ref_t=list(cal.n_ref), theta0_t=list(cal.theta0), n_seeds=cal.rows[-1].n_seeds,
                K_full=full.K, n_ref_full=full.n_ref, theta0_full=full.theta0)


def engine_defaults() -> dict:
    """The engine's keyword defaults (keeper.scem.property1's reading of its signature)."""
    return {k: v for k, v in P1._engine_config({}).items() if k not in ("witness_cred", "return_records")}


def horizon_A(policy: str) -> float:
    """The closure age A of a policy, as the engine builds it."""
    cfg = P1._engine_config(POLICIES[policy])
    return release_policy(cfg["policy"], cfg["N"], cfg["phi"], cfg["q_quantile"], cfg["quarantine_D"]).A


def policy_table(names) -> dict:
    return {p: dict(POLICIES[p], A=horizon_A(p)) for p in names}


# ------------------------------------------------------------------ E1a metrics


def _mean(x, mask):
    return float(np.mean(x[mask])) if mask.any() else None


def released_share(out: dict, snap, age: int, K: int):
    """Mean released share of the surviving mass at age `age` (clarification 6)."""
    rec = out["records"]
    row = snap + age - 1
    idx = np.flatnonzero(row <= K - 1)
    omega = rec["omega_by_snap"][row[idx], idx]
    total = omega + rec["pi_by_snap"][row[idx], idx]
    live = total > 0
    return float(np.mean(omega[live] / total[live])) if live.any() else None


def escrow_metrics(stream: Stream, out: dict) -> dict:
    """E1a's metrics of an engine run on the clean stream. A rating is flagged as in stage_b.engine_metrics: its sender
    receives at least two negative ratings after it, and more negative than positive ones. The future-only rule keeps
    the flagged ratings whose sender had received no negative rating at or before the rating's snapshot. Honest mass
    revoked: ratings of senders never flagged (clarification 5). Kept weight is the engine's effective_weight."""
    edges, sign, snap, K = stream.edges, stream.sign, stream.snap, stream.K
    cneg, cpos = stage_b.incoming_cum(WC.stream_graph(*stream.clean), sign, stream.n, K)
    u, end = edges[:, 0], np.full_like(snap, K - 1)
    neg_after, pos_after = stage_b.window(cneg, u, snap, end), stage_b.window(cpos, u, snap, end)
    flagged = (neg_after >= 2) & (neg_after > pos_after)
    future_only = flagged & (cneg[u, snap] == 0)
    lost = 1.0 - out["effective_weight"]
    ever_flagged = np.zeros(stream.n, dtype=bool)
    ever_flagged[u[flagged]] = True
    honest = ~ever_flagged[u]
    count = np.bincount(u[honest], minlength=stream.n)
    per_sender = np.bincount(u[honest], weights=lost[honest], minlength=stream.n)[count > 0] / count[count > 0]
    return dict(n_ratings=len(sign), n_flagged=int(flagged.sum()), n_future_only=int(future_only.sum()),
                n_honest_ratings=int(honest.sum()), n_honest_senders=int((count > 0).sum()),
                silent_sender_revocation=_mean(lost, flagged),
                silent_sender_revocation_future_only=_mean(lost, future_only),
                honest_mass_revoked=_mean(lost, honest),
                honest_mass_revoked_sender_mean=float(per_sender.mean()) if len(per_sender) else None,
                all_retained=float(np.mean(out["effective_weight"])),
                latency_age1=released_share(out, snap, 1, K), latency_age4=released_share(out, snap, 4, K),
                max_conservation_err=float(out["max_conservation_err"]),
                monotone_verified=bool(out["monotone_verified"]))


def study_e1a(stream: Stream) -> dict:
    levels = {"off": None, **{v: witness_levels(stream, *stream.clean, stream.n, v) for v in CAP_UNITS}}
    metrics = {p: {mb: escrow_metrics(stream, run_engine(stream, *stream.clean, stream.n, lv, p))
                   for mb, lv in levels.items()} for p in POLICIES}
    return dict(stream=stream.record(), policies=policy_table(POLICIES), metrics=metrics,
                calibration={v: calibration_record(stream, v) for v in CAP_UNITS})


# ------------------------------------------------------------------ victim records and batches


def victim_records(stream: Stream) -> np.ndarray:
    """Warnings W -> T with T in the pick_targets cohort (distrust majority, at most 3 distrust in-edges)."""
    targets = C.pick_targets(WC.stream_graph(*stream.clean))[0]
    return WA.warnings_about(stream.edges, stream.sign, targets)


def attackable(stream: Stream, records, lag: int) -> tuple:
    """Ruling R34 as inject_adversarial_witnesses applies it: (records attacked at this lag, records skipped because
    t_k + lag <= t_seed + 1, records skipped because t_k + lag >= K)."""
    records = np.asarray(records, dtype=np.int64)
    t_b = stream.snap[records] + lag
    early, late = t_b <= stream.t_seed + 1, t_b >= stream.K
    return records[~early & ~late], records[early], records[late]


def batch_cap(k: int, n_ref: int) -> int:
    """The most records a batch holds with k x records <= 2% of n_ref."""
    cap = n_ref // (BATCH_SHARE_DENOM * k)
    if cap < 1:
        raise ValueError(f"k = {k} Sybils exceed 2% of n_ref = {n_ref} even for one record")
    return cap


def make_batches(records, warners, cap: int) -> list:
    """First fit in the given order: a record joins the first batch that holds no record of its warner and fewer
    than cap records, else it opens a new batch (clarification 1)."""
    batches, seen = [], []
    for rec, warner in zip(np.asarray(records).tolist(), np.asarray(warners).tolist()):
        for batch, ws in zip(batches, seen):
            if warner not in ws and len(batch) < cap:
                batch.append(rec)
                ws.add(warner)
                break
        else:
            batches.append([rec])
            seen.append({warner})
    return [np.asarray(b, dtype=np.int64) for b in batches]


def cell_plan(stream: Stream, records, k: int, lag: int) -> dict:
    attacked, early, late = attackable(stream, records, lag)
    cap = batch_cap(k, stream.n_ref)
    return dict(cap=cap, batches=make_batches(attacked, stream.edges[attacked, 0], cap), skipped_seed_window=early,
                skipped_beyond_stream=late)


def plans_for(stream: Stream, records, cells) -> list:
    cache = {}
    for c in cells:
        if (c.k, c.lag) not in cache:
            cache[c.k, c.lag] = cell_plan(stream, records, c.k, c.lag)
    return [cache[c.k, c.lag] for c in cells]


def eligible(t_k, lag: int, A: float, L_f: int, K: int) -> np.ndarray:
    """H5's eligibility of victim records: lag + L_f >= A_P and t_k + lag + L_f <= K - 1."""
    t_k = np.asarray(t_k)
    return (lag + L_f >= A) & (t_k + lag + L_f <= K - 1)


def single_plan(stream: Stream, records, lag: int) -> dict:
    """The single-record plan at one lag: one batch per attacked record that is eligible under some committing
    policy; the attacked records eligible under none are listed, not run."""
    attacked, early, late = attackable(stream, records, lag)
    L_f = engine_defaults()["L_f"]
    keep = np.any([eligible(stream.snap[attacked], lag, horizon_A(p), L_f, stream.K) for p in COMMITTING], axis=0)
    return dict(cap=1, batches=[attacked[i:i + 1] for i in np.flatnonzero(keep)], skipped_seed_window=early,
                skipped_beyond_stream=late, not_eligible=attacked[~keep])


def cell_record(index: int, cell, plan: dict) -> dict:
    extra = {"not_eligible": plan["not_eligible"]} if "not_eligible" in plan else {}
    return dict(index=index, **cell._asdict(), cap=plan["cap"], n_batches=len(plan["batches"]),
                n_records=int(sum(len(b) for b in plan["batches"])),
                skipped_seed_window=plan["skipped_seed_window"], skipped_beyond_stream=plan["skipped_beyond_stream"],
                **extra)


def shard_units(n_batches, shard: tuple) -> tuple:
    """(every unit (cell, batch) in order, the units of shard (i, n): i, i + n, ...)."""
    units = [(c, b) for c, nb in enumerate(n_batches) for b in range(nb)]
    index, count = shard
    return units, units[index::count]


def inject(stream: Stream, base_levels, cell, cell_index: int, batch_index: int, records):
    rng = np.random.default_rng((ENTROPY, stream.net_index, batch_index, cell_index))
    att = WA.inject_adversarial_witnesses(*stream.clean, stream.n, records, cell.k, cell.lag, cell.g, rng,
                                          base_levels, stream.K, t_seed=stream.t_seed, sybil_ring=cell.ring,
                                          kappa_farming=cell.farming)
    if not np.array_equal(att.victims, records):
        raise RuntimeError("the injector skipped a batched record; batches must hold attackable records only")
    return att


def comparator_stream(att) -> tuple:
    """The attacked stream without its bad-mouthing ratings (Section 2.3's comparator). The injected rows follow the
    original ones, so every original record keeps its id."""
    keep = np.ones(len(att.sign), dtype=bool)
    keep[att.badmouth_edge_ids.reshape(-1)] = False
    return att.edges[keep], att.sign[keep], att.snap[keep]


# ------------------------------------------------------------------ E1b, E1c


def attack_unit(stream: Stream, base_levels, clean: dict, cell, cell_index: int, batch_index: int, records) -> dict:
    """One attacked stream of E1b: per victim record and M-B setting, the revoked share under attack, on the
    comparator (with the attacked stream's levels) and on the clean stream, and property1_bound."""
    att = inject(stream, base_levels, cell, cell_index, batch_index, records)
    levels = witness_levels(stream, att.edges, att.sign, att.snap, att.n_nodes)
    comp = comparator_stream(att)
    unit = dict(cell=cell_index, batch=batch_index, victims=records, warners=stream.edges[records, 0],
                t_k=stream.snap[records], endorsers=att.endorsers, pool_size=att.pool_size,
                honest_tier_changed=int((levels[:, :stream.n] != base_levels).any(axis=0).sum()))
    for mb, lv in (("off", None), ("on", levels)):
        attacked = run_engine(stream, att.edges, att.sign, att.snap, att.n_nodes, lv, E1B_POLICY)
        comparator = run_engine(stream, *comp, att.n_nodes, lv, E1B_POLICY)
        bound = P1.property1_bound(att.edges, att.sign, att.snap, att.badmouth_edge_ids.reshape(-1),
                                   attacked["records"]["witness_kappa"], K_snap=stream.K, t_seed=stream.t_seed,
                                   **POLICIES[E1B_POLICY])
        unit[mb] = dict(attack=revoked_share(attacked)[records], comparator=revoked_share(comparator)[records],
                        clean=clean[mb][records], bound=bound[records],
                        max_conservation_err=max(attacked["max_conservation_err"],
                                                 comparator["max_conservation_err"]),
                        monotone_verified=attacked["monotone_verified"] and comparator["monotone_verified"])
    return unit


def study_e1b(stream: Stream, shard=(0, 1), cells=E1B_CELLS) -> dict:
    records = victim_records(stream)
    base = witness_levels(stream, *stream.clean, stream.n)
    clean = {mb: revoked_share(run_engine(stream, *stream.clean, stream.n, lv, E1B_POLICY))
             for mb, lv in (("off", None), ("on", base))}
    plans = plans_for(stream, records, cells)
    units, mine = shard_units([len(p["batches"]) for p in plans], shard)
    return dict(stream=stream.record(), policy=E1B_POLICY, policies=policy_table([E1B_POLICY]),
                n_victim_records=len(records),
                cells=[cell_record(i, c, p) for i, (c, p) in enumerate(zip(cells, plans))], n_units=len(units),
                units=[attack_unit(stream, base, clean, cells[c], c, b, plans[c]["batches"][b]) for c, b in mine],
                calibration={"capped": calibration_record(stream, "capped")})


def study_e1c(stream: Stream, lags, shard=(0, 1), policies=E1C_POLICIES, single=False) -> dict:
    """H5's window study: M-B off, k = 5, g = 0, ring off; per unit (lag, batch), every policy on one attacked
    stream. single: one eligible record per batch (single_plan), as e1c_single and e1d_h5_single run it."""
    records = victim_records(stream)
    base = witness_levels(stream, *stream.clean, stream.n)       # the injector's input; g = 0 draws no endorser
    clean = {p: revoked_share(run_engine(stream, *stream.clean, stream.n, None, p)) for p in policies}
    cells = [Cell(E1C_K, E1C_G, lag, False) for lag in lags]
    plans = [single_plan(stream, records, c.lag) for c in cells] if single else plans_for(stream, records, cells)
    units, mine = shard_units([len(p["batches"]) for p in plans], shard)
    out = []
    for c, b in mine:
        recs = plans[c]["batches"][b]
        att = inject(stream, base, cells[c], c, b, recs)
        out.append(dict(cell=c, batch=b, lag=cells[c].lag, victims=recs, t_k=stream.snap[recs],
                        attack={p: revoked_share(run_engine(stream, att.edges, att.sign, att.snap, att.n_nodes, None,
                                                            p))[recs] for p in policies},
                        clean={p: clean[p][recs] for p in policies}))
    return dict(stream=stream.record(), lags=list(lags), policies=policy_table(policies),
                L_f=engine_defaults()["L_f"], n_victim_records=len(records),
                cells=[cell_record(i, c, p) for i, (c, p) in enumerate(zip(cells, plans))], n_units=len(units),
                units=out)


# ------------------------------------------------------------------ E1d


def study_e1d(D) -> dict:
    cuts = {}
    for cut in CUT_T_SEED:
        s = cut_stream(D, cut)
        levels = {"off": None, "capped": witness_levels(s, *s.clean, s.n)}
        days = float(s.time.max() - s.time.min()) / 86400.0 / s.K
        cuts[str(cut)] = dict(
            stream=s.record(), days_per_snapshot=days,
            policies={p: dict(row, horizon_days=row["A"] * days) for p, row in policy_table(E1D_POLICIES).items()},
            metrics={p: {mb: escrow_metrics(s, run_engine(s, *s.clean, s.n, lv, p)) for mb, lv in levels.items()}
                     for p in E1D_POLICIES},
            calibration={"capped": calibration_record(s, "capped")})
    return dict(cuts=cuts)


def study_e1d_h5(D, shard, single=False) -> dict:
    stream = cut_stream(D, H5_CUT)
    return dict(study_e1c(stream, tuple(H5_LAG_SCALE * lag for lag in E1B_LAGS), shard, single=single), cut=H5_CUT)


def run_study(study: str, ds: str, shard: tuple) -> tuple:
    """(D, the study's body)."""
    single = study.endswith("_single")
    if study in ("e1d", "e1d_h5", "e1d_h5_single"):
        D = C.load(ds)
        return D, study_e1d(D) if study == "e1d" else study_e1d_h5(D, shard, single)
    D, stream = training_stream(ds)
    if study == "e1a":
        return D, study_e1a(stream)
    if study == "e1b":
        return D, study_e1b(stream, shard)
    return D, study_e1c(stream, E1B_LAGS, shard, single=single)


def parse_shard(text: str) -> tuple:
    try:
        index, count = (int(x) for x in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard must be I/N, got {text!r}") from None
    if not 0 <= index < count:
        raise ValueError(f"--shard {text}: need 0 <= I < N")
    return index, count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.scem.e1_runner", description=__doc__.splitlines()[0])
    ap.add_argument("study", choices=STUDIES)
    ap.add_argument("--ds", required=True, choices=ps.NETWORKS)
    ap.add_argument("--out", required=True, help="absolute path of the output JSON")
    ap.add_argument("--shard", default="0/1", help=f"I/N: the units I, I + N, ... ({', '.join(SHARDED)})")
    ap.add_argument("--prereg", default=os.path.join(C.ROOT, ps.PREREG), help="the protocol file whose sha256 is kept")
    ap.add_argument("--force", action="store_true", help="replace an existing output")
    a = ap.parse_args(argv)
    try:
        shard = parse_shard(a.shard)
    except ValueError as error:
        ap.error(str(error))
    if shard != (0, 1) and a.study not in SHARDED:
        ap.error(f"{a.study} has no shards")
    if not os.path.isabs(a.out):
        ap.error("--out must be an absolute path")
    if C.DEV.type != "cpu":
        ap.error("E1 runs on the CPU only; set CUDA_VISIBLE_DEVICES=''")
    try:
        path = C.result_path(a.out, a.force)          # refuse an existing output before the work
    except FileExistsError as error:
        ap.error(str(error))
    prereg = preregistration(a.prereg)
    t0 = time.time()
    D, body = run_study(a.study, a.ds, shard)
    doc = dict(study=a.study, ds=a.ds, shard=dict(index=shard[0], count=shard[1]), engine_defaults=engine_defaults(),
               preregistration=prereg, **body, runtime_s=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    C.dump(dict(doc, provenance=dict(C.provenance(D), **git_state())), path, D, force=a.force)
    print(f"wrote {path} ({doc['runtime_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
