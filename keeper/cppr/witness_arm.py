"""E7's witness-laundering arm (Task 4.7; protocol docs/preregistration.md Sections 2.1 and 2.9): adversarial witnesses
against the honest warners of the CPPR direct targets, on a stream without deletions, with the rating-level backbone.

  python -m keeper.cppr.witness_arm --ds DS --seed S --stream {clean,g0,g1} --mb {on,off} --epochs 150 \
         --prereg docs/preregistration.md --out ABS.json [--force]

Targets: the edge-level direct targets that the CPPR driver draws for the seed (its workload on the training graph,
direct_targets; keeper.analysis.cppr_runs.seed_targets replays the same draw). Victims: their warnings, the negative
training ratings whose receiver is a direct target (keeper.scem.witness_attack.warnings_about); the warnings' senders
are the honest warners. The stream is the network's training graph (snapshots 0..tt-1); nothing is erased.

Attack (streams g0 and g1): keeper.scem.witness_attack.inject_adversarial_witnesses with K_SYBILS = 5 fresh Sybils per
warning that bad-mouth its warner at LAG = 1, a positive Sybil ring, and g attack edges per warning from honest
accounts whose causal capped tier at the attack-edge snapshot is >= 2. Those tiers are the clean stream's causal
capped levels under the calibration frozen on it (keeper.scem.e2_streams.stream_levels), and the endorsers are drawn
from default_rng((20261027, seed, ARM_INDEX, g)). The timing is that function's (rulings R34 and R39): attack edges and
ring at max(t_k + lag - 1, t_seed + 1), bad-mouthing at t_k + lag; a warning whose bad-mouthing would fall at or before
t_seed + 1, or at or after snapshot tt, is skipped and recorded. The injected rows take the last timestamp of their
snapshot and negative row ids, and the stream is sorted stably by snapshot (as keeper.scem.e2_streams), so the original
rows keep their order and lead every snapshot. Stream clean is the training graph unchanged: the twin of both attacks.

Model: keeper.cppr.rating_backbone.train_rating_model, the backbone of the CPPR runs with --backbone rating_scem. It is
the rating-level LocalGDTE-SCEM built through core.make_model with the run's seed and gated by CPPR's gate (norm count)
with capped BTP tiers on its own stream, under the calibration frozen on the honest training graph (the driver's tiers
of an injected graph). M-B on: the witness weights of the stream's causal capped levels; M-B off: none (the stored
witness rule). Gate and M-B use the network's capped eps, KEEPER's cap and guard (keeper.cluster.protocol_settings).

Scores: the direct targets' proxies (core.Panel on the clean training graph, n_probe 256 and seed 1234, the driver's
panel) and the test metrics on the held-out snapshot. The gain, proxy under attack minus proxy on the clean twin with
the same M-B setting, needs two runs and is computed by keeper.analysis.e7_analysis.

Output: one JSON per network, seed, stream and M-B setting through core.dump (atomic; an existing file is replaced only
with --force): the plan and its digest, the attack's endorsers and timing, the stream's digest, the Sybils' gate tiers,
the M-B record, the model's settings and learned phi, the scores, the determinism warnings, the protocol's sha256 and
the provenance with the git state.
"""
import argparse
import hashlib
import os
import sys
import time
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

from keeper.analysis import cppr_runs as R  # noqa: E402
from keeper.analysis.provenance import git_state  # noqa: E402
from keeper.cluster import protocol_settings as ps  # noqa: E402
from keeper.cppr import rating_backbone as RB  # noqa: E402
from keeper.cppr.protocol import ATTACK_RNG_ENTROPY, CredibilityTiers, preregistration  # noqa: E402
from keeper.credibility import tiers as CT  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.scem import e2_driver as E2  # noqa: E402
from keeper.scem import e2_streams as E  # noqa: E402
from keeper.scem import witness_attack as WA  # noqa: E402
from keeper.scem.witness_cred import level_weights  # noqa: E402

K_SYBILS, LAG = 5, 1
ARM_INDEX = 7                  # endorser draws: default_rng((ATTACK_RNG_ENTROPY, seed, ARM_INDEX, g))
GATE_NORM = "count"            # CPPR's gate (Eq. 15), the driver's default
PANEL_PROBES, PANEL_SEED = 256, 1234


def direct_targets(Tr, seed: int) -> np.ndarray:
    """The edge-level direct targets of the CPPR driver's workload for ``seed`` on the training graph Tr."""
    cohort, neg, _ = C.pick_targets(Tr, max_neg=R.MAX_NEG, strict=True)
    honest_rng = np.random.default_rng(int(1e6 * R.RHO) + 17 * seed)
    _, edge_t, _ = R.workload_function()(Tr, cohort, neg, R.N_EDGE, R.N_ACC, R.RHO, R.TARGET_SEED + seed, honest_rng)
    return np.asarray(edge_t, dtype=np.int64)


def endorser_rng(seed: int, g: int) -> np.random.Generator:
    return np.random.default_rng((ATTACK_RNG_ENTROPY, int(seed), ARM_INDEX, int(g)))


def attack_stream(Tr, targets, base_levels, seed: int, g: int, tt: int) -> tuple:
    """(attacked training graph, keeper.scem.witness_attack.InjectedStream) of the warnings about ``targets`` in the
    clean training graph Tr; ``base_levels`` are the clean stream's causal tiers, from which endorsers are drawn."""
    edges, sign, snap = E.signed(Tr)
    inj = WA.inject_adversarial_witnesses(edges, sign, snap, Tr.n, WA.warnings_about(edges, sign, targets), K_SYBILS,
                                          LAG, g, endorser_rng(seed, g), base_levels, tt, sybil_ring=True)
    new_snap = inj.snap[len(edges):]
    stamp = np.array([Tr.time[Tr.snap == s].max() if (Tr.snap == s).any() else 0 for s in new_snap], dtype=np.int64)
    G = C.Graph(inj.edges, np.concatenate([Tr.labels, S.one_hot_from_sign(inj.sign[len(edges):])]),
                np.concatenate([Tr.time, stamp]), inj.snap, inj.n_nodes,
                np.concatenate([Tr.rid, -1 - np.arange(len(new_snap), dtype=np.int64)]))
    return G.subset(np.argsort(G.snap, kind="mergesort")), inj


def _sha256(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode() + a.tobytes())
    return h.hexdigest()


def _triples(Tr, rows) -> list:
    """[warner, target, snapshot] of training rows."""
    rows = np.asarray(rows, dtype=np.int64)
    return np.stack([Tr.edges[rows, 0], Tr.edges[rows, 1], Tr.snap[rows]], axis=1).tolist() if len(rows) else []


def attack_record(Tr, inj, seed: int, g: int) -> dict:
    """What the attack injected, per attacked warning in record order, and the warnings it skipped (ruling R34)."""
    return dict(g=g, k=K_SYBILS, lag=LAG, sybil_ring=True, rng=[ATTACK_RNG_ENTROPY, int(seed), ARM_INDEX, int(g)],
                victims=_triples(Tr, inj.victims), t_attack=inj.t_attack.tolist(), t_badmouth=inj.t_badmouth.tolist(),
                endorsers=inj.endorsers.tolist(), endorser_levels=inj.endorser_levels.tolist(),
                pool_size=inj.pool_size.tolist(), skipped_seed_window=_triples(Tr, inj.skipped_seed_window),
                skipped_beyond_stream=_triples(Tr, inj.skipped_beyond_stream),
                attacked_targets=sorted({int(v) for v in Tr.edges[inj.victims, 1]}),
                n_sybils=int(inj.sybil_ids.size), n_injected=int(len(inj.edges) - len(Tr.edges)))


def run(ds: str, seed: int, stream: str, mb: str, epochs: int, out: str, prereg: str, force: bool = False) -> dict:
    """One witness-arm run; writes its JSON and returns the record."""
    t0 = time.time()
    if stream not in ps.E7_WITNESS_STREAMS or mb not in ps.E7_MB:
        raise ValueError(f"stream must be one of {tuple(ps.E7_WITNESS_STREAMS)} and mb one of {ps.E7_MB}")
    path = C.result_path(out, force)              # refuse an existing result before the work, not after it
    prereg_info = preregistration(prereg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(ds)
    mem = C.apply_memory_profile(D)
    G, tt, eps, g = C.full_graph(D), D.tt, ps.EPS_CAPPED[ds], ps.E7_WITNESS_STREAMS[stream]
    Tr = C.train_split(G, tt)
    targets = direct_targets(Tr, seed)
    edges, sign, _ = E.signed(Tr)
    victims = WA.warnings_about(edges, sign, targets)
    clean_levels = E.stream_levels(Tr, Tr, eps, tt)
    Tr_run, inj = (Tr, None) if g is None else attack_stream(Tr, targets, clean_levels, seed, g, tt)
    tier_fn = CredibilityTiers("btp", CT.calibrate_btp(Tr, D.n, eps, edge_cap_units=ps.EDGE_CAP_UNITS), ps.NEG_FROM)
    cred = tier_fn(Tr_run, Tr_run.n)
    levels = clean_levels if g is None else E.stream_levels(Tr_run, Tr, eps, tt)
    mb_info = None
    if mb == "on":
        mb_info = dict(E2.mb_record(levels, clean_levels, Tr, eps, None, None),
                       honest_levels_changed=int((levels[:, :Tr.n] != clean_levels).sum()))
    with warnings.catch_warnings(record=True) as caught:      # protocol 2.1: log determinism warnings
        warnings.simplefilter("always")
        t1 = time.time()
        m, settings = RB.train_rating_model(D, Tr_run, seed, tt, cred, level_weights(levels) if mb == "on" else None,
                                            epochs, gate_norm=GATE_NORM)
        train_s = time.time() - t1
        z = C.embed(m, Tr_run, tt)
    determinism = sorted({str(w.message) for w in caught if "determinis" in str(w.message)})
    for message in determinism:
        print(f"determinism warning: {message}", file=sys.stderr)
    panel = C.Panel(Tr, n_probe=PANEL_PROBES, seed=PANEL_SEED)
    test = C.test_metrics(m, z, D, G)
    record = dict(
        experiment="E7", arm="witness", ds=ds, seed=seed, model_seed=seed, stream=stream, g=g, mb=mb, epochs=epochs,
        backbone=RB.RATING, model=RB.SCEM_MODEL, gate_norm=GATE_NORM, settings=settings, learned_phi=RB.learned_phi(m),
        plan=dict(targets=targets.tolist(), victims=_triples(Tr, victims), sha256=_sha256(targets, victims)),
        attack=None if inj is None else attack_record(Tr, inj, seed, g),
        stream_info=dict(n_nodes=int(Tr_run.n), n_rows=int(len(Tr_run.edges)),
                         sha256=_sha256(Tr_run.edges, Tr_run.labels, Tr_run.snap)),
        gate=dict(method="btp", eps=float(eps), edge_cap_units=ps.EDGE_CAP_UNITS, neg_from=ps.NEG_FROM,
                  sybil_tier_counts=np.bincount(cred[Tr.n:], minlength=len(CT.TIER_WEIGHTS)).tolist()),
        mb_info=mb_info, targets=targets.tolist(), proxy=panel.reputation(m, z, targets).tolist(),
        panel=dict(n_probe=int(len(panel.sources)), sources_sha256=_sha256(panel.sources)),
        test_metrics=test, test_mcc=test["t1"]["mcc"], memory_profile=mem, determinism_warnings=determinism,
        prereg=prereg_info, train_s=round(train_s, 1), runtime_s=round(time.time() - t0, 1))
    C.dump(dict(record, provenance=dict(C.provenance(D), **git_state())), path, D, force=force)
    print(f"[E7 witness {ds} s={seed} {stream} M-B {mb}] test MCC {record['test_mcc']} "
          f"({record['runtime_s']} s); wrote {path}")
    return record


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m keeper.cppr.witness_arm", description="One E7 witness-arm run.")
    ap.add_argument("--ds", required=True, choices=ps.NETWORKS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--stream", required=True, choices=tuple(ps.E7_WITNESS_STREAMS))
    ap.add_argument("--mb", required=True, choices=ps.E7_MB, help="M-B witness weights on or off")
    ap.add_argument("--epochs", type=int, default=ps.EPOCHS)
    ap.add_argument("--prereg", required=True, help="the protocol file; its sha256 is recorded")
    ap.add_argument("--out", required=True, help="the result JSON (absolute, or a name under results/laundering/)")
    ap.add_argument("--force", action="store_true", help="replace an existing result (or set KEEPER_OVERWRITE=1)")
    return ap


def main(argv=None) -> None:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        run(a.ds, a.seed, a.stream, a.mb, a.epochs, a.out, a.prereg, a.force)
    except (FileExistsError, FileNotFoundError) as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
