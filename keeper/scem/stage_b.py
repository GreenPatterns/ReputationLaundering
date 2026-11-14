"""Stage B evaluation of the user's ORIGINAL SCEM code: only the rows that code can compute.

NOTHING HERE CHANGES SCEM. Two pieces of the user's code are used as written:
  * keeper/scem/scem.py (updated/scem.py with the four out-of-place index_add lines the user approved
    on 2026-10-17; tests/test_scem_equivalence.py shows the forward pass unchanged; the Task 3.3 options,
    upgrade M-B and the release policies, keep their defaults here, under which tests/test_scem_upgrades.py
    regenerates the stored engine records):
      - run_scem_streaming_weights_np: the fixed-parameter NumPy streaming engine
        (defaults N=2, phi=0.5, so A = 13.28 snapshots), returning per-rating committed
        (omega), provisional (pi) and revoked mass, per-node hazards and a conservation check;
      - verify_symbolic_example: the silent-sender / symbolic trace.
  * updated/archive_old_versions/local_model.py: the user's LocalGDTE with SCEM as its
    temporal encoder (it adds SCEM's auxiliary loss), trained with laundering.core.train.

Rows (user's decision 2026-10-17: "only rows your code computes"):
  engine  per dataset; deterministic, because the engine has no learned parameters and no seed
          - default (A = 13.28): raw / matured coverage, attenuation, retention, conservation error
          - Erlang order at matched mean delay: (N, phi) = (1, 0.25), (2, 0.50), (3, 0.75)
          - components: V2 only (cap_c = 0), V1 only (c_self = 0), no supersession (impulse = 0)
          - rolling landmarks t* in {7, 10, 14}: AUROC / AUPRC of the engine hazard, attenuation
  trace   verify_symbolic_example() (silent-sender block)
  model   per dataset and seed: LocalGDTE+SCEM trained for 150 epochs; ordinary MCC and AUROC
          (laundering.core.test_metrics, Task 1, p < 0.5), ECE, peak GPU memory, latency
Not computed (need code the user does not have): burst protocol and burst MCC, fixed-delay
buffers, unescrowed / durable-only / hazard-gated readouts, wall-clock horizons, temporal baselines.

Definitions. [paper] = stated in main_scem.tex. [proposed] = defined neither in the paper nor
in the user's code; the user must confirm these before any result is used.
  y_fut(u; t*)     [paper, l.449] 1{ d-_in(u) over (t*, 20] >= 2 and d-_in > d+_in over (t*, 20] }
  rolling score    [proposed] the engine hazard lambda_u summed over its two channels, from the
                   engine run on snapshots 0..t* only
  rolling pop.     [proposed] accounts with >= 1 outgoing rating in 0..t* and >= 1 incoming rating
                   in (t*, 20], so that y_fut is observable
  flagged rating   [proposed; y_fut with t* = the rating's snapshot s_k] its sender receives >= 2
                   negative ratings, and more negative than positive, in (s_k, 20]
  attenuation      [proposed; "attenuates X% of future-distrust sender mass"] mean revoked mass
                   1 - omega - pi of flagged ratings at the end of snapshot 20
  all retained     [proposed] mean omega + pi over all training ratings
  controv. retained[proposed] mean omega + pi over ratings whose sender receives >= 1 negative
                   rating in (s_k, 20] but is not flagged
  raw coverage     [proposed] share of flagged ratings whose sender receives >= 1 negative rating
                   in (s_k, s_k + A]; matured coverage: in (s_k, s_k + A - L_f]
  rolling atten.   [proposed] mean revoked mass, after the prefix run, of prefix ratings whose
                   sender has y_fut(u; t*) = 1
  ECE              [proposed] 10 equal-width bins of P(distrust) on the Task 1 held-out edges
  memory, latency  [proposed] peak torch.cuda allocation during training / inference; inference
                   wall time divided by the number of training ratings
Training snapshots are 0..20 on all three datasets (T = 22, held-out snapshot 21; checked
2026-10-17), so the paper's window (t*, 20] lies inside the training stream.

Usage (from the project root; outputs go to results/scem/stage_b/, or to the folder --out DIR names, and an
existing record is replaced only with --force or KEEPER_OVERWRITE=1):
  python -m keeper.scem.stage_b engine --ds bitcoinalpha   # CPU
  python -m keeper.scem.stage_b trace                      # CPU
  python -m keeper.scem.stage_b equiv --ds bitcoinalpha --seed 0 --original FILE  # GPU; FILE: git show
                                                  # pre-revision-2026-10-26:thecode/scem.py > FILE
  python -m keeper.scem.stage_b model --ds bitcoinalpha --seed 0  # GPU
  python -m keeper.scem.stage_b aggregate
"""
import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402

from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.scem import scem as SC  # noqa: E402  (the user's scem.py with the four approved index_add lines)

OUT = os.path.join(ROOT, "results", "scem", "stage_b")   # scem_eval/results_stageB/ before 2026-10-26
LANDMARKS = (7, 10, 14)
ENGINE_CONFIGS = {                     # name -> keyword arguments of the user's engine
    "default": dict(N=2, phi=0.5),
    "erlang_N1": dict(N=1, phi=0.25),
    "erlang_N3": dict(N=3, phi=0.75),
    "v2_only": dict(N=2, phi=0.5, cap_c=0.0),
    "v1_only": dict(N=2, phi=0.5, c_self=0.0),
    "no_supersession": dict(N=2, phi=0.5, supersession_impulse=0.0),
}
ENGINE_LF = 1                          # the engine's default L_f (snapshots)


def record_path(name, force=False):
    """Path of a record in OUT. An existing record is evidence: it is replaced only with force (the CLI's
    --force) or KEEPER_OVERWRITE=1 (keeper.laundering.core.overwrite_allowed)."""
    path = os.path.join(OUT, name)
    if os.path.exists(path) and not C.overwrite_allowed(force):
        raise FileExistsError(f"{path} exists; pass --force or set {C.OVERWRITE_ENV}=1 to overwrite it")
    return path


def save(name, obj, force=False):
    """Write a record to OUT atomically (keeper.laundering.core.write_atomic): complete or absent, and an existing
    record, even one that another writer created meanwhile, is replaced only with force or KEEPER_OVERWRITE=1."""
    path = record_path(name, force)
    os.makedirs(OUT, exist_ok=True)
    C.write_atomic(path, lambda f: json.dump(obj, f, indent=1, default=float), replace=C.overwrite_allowed(force))
    print("wrote", path)


def load_stream(ds):
    """The user's own loader and training split: snapshots 0..tt-1 train, tt held out."""
    D = C.load(ds)
    G = C.full_graph(D)
    Tr = C.train_split(G, D.tt)
    sign = 1 - 2 * S.class_ids(Tr.labels)            # +1 trust, -1 distrust
    return D, G, Tr, sign


def incoming_cum(Tr, sign, n_nodes, n_snap):
    """cum_neg[v, s] / cum_pos[v, s]: negative / positive ratings v received in snapshots 0..s."""
    neg = np.zeros((n_nodes, n_snap), np.int64)
    pos = np.zeros((n_nodes, n_snap), np.int64)
    dst = Tr.edges[:, 1]
    np.add.at(neg, (dst[sign < 0], Tr.snap[sign < 0]), 1)
    np.add.at(pos, (dst[sign > 0], Tr.snap[sign > 0]), 1)
    return np.cumsum(neg, 1), np.cumsum(pos, 1)


def window(cum, nodes, lo, hi):
    """Counts in snapshots (lo, hi], clipped to the stream; lo, hi integer arrays."""
    last = cum.shape[1] - 1
    hi = np.minimum(hi, last)
    out = cum[nodes, hi] - cum[nodes, np.minimum(lo, last)]
    return np.where(hi > lo, out, 0)


def engine_metrics(Tr, sign, n_nodes, cfg):
    """One run of the user's engine on the full training stream, and the escrow metrics."""
    K = int(Tr.snap.max()) + 1
    t0 = time.time()
    res = SC.run_scem_streaming_weights_np(Tr.edges, sign, Tr.snap, n_nodes, K_snap=K, **cfg)
    secs = time.time() - t0
    A = SC.erlang_horizon_A(cfg["N"], cfg["phi"])
    cneg, cpos = incoming_cum(Tr, sign, n_nodes, K)
    u, s = Tr.edges[:, 0], Tr.snap.astype(np.int64)
    end = np.full_like(s, K - 1)
    neg_after = window(cneg, u, s, end)
    pos_after = window(cpos, u, s, end)
    flagged = (neg_after >= 2) & (neg_after > pos_after)
    controversial = (neg_after >= 1) & ~flagged
    raw = window(cneg, u, s, s + int(math.floor(A))) >= 1
    mat = window(cneg, u, s, s + int(math.floor(A - ENGINE_LF))) >= 1
    kept = res["effective_weight"]                   # the engine's clip(omega + pi, 0, 1)

    def mean(x, m):
        return None if not m.any() else float(np.mean(x[m]))
    return dict(config=cfg, A=float(A), seconds=round(secs, 1), n_ratings=int(len(sign)),
                n_flagged=int(flagged.sum()), n_controversial=int(controversial.sum()),
                raw_coverage=mean(raw.astype(float), flagged),
                matured_coverage=mean(mat.astype(float), flagged),
                attenuation=mean(1.0 - kept, flagged),
                all_retained=float(np.mean(kept)),
                controversial_retained=mean(kept, controversial),
                max_conservation_err=float(res["max_conservation_err"]),
                monotone_verified=bool(res["monotone_verified"]))


def rolling(Tr, sign, n_nodes):
    """Rolling landmarks with the default engine run on the prefix 0..t* only."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    K = int(Tr.snap.max()) + 1
    cneg, cpos = incoming_cum(Tr, sign, n_nodes, K)
    out = {}
    for ts in LANDMARKS:
        keep = Tr.snap <= ts
        P = Tr.subset(keep)
        res = SC.run_scem_streaming_weights_np(P.edges, sign[keep], P.snap, n_nodes, K_snap=ts + 1,
                                               **ENGINE_CONFIGS["default"])
        score = res["lambda_nodes"].sum(1)
        nodes = np.arange(n_nodes)
        lo = np.full(n_nodes, ts)
        hi = np.full(n_nodes, K - 1)
        neg_f, pos_f = window(cneg, nodes, lo, hi), window(cpos, nodes, lo, hi)
        y = (neg_f >= 2) & (neg_f > pos_f)
        emitted = np.zeros(n_nodes, bool)
        emitted[P.edges[:, 0]] = True
        pop = emitted & ((neg_f + pos_f) >= 1)
        yp, sp = y[pop], score[pop]
        kept = res["effective_weight"]
        fd = y[P.edges[:, 0]]
        out[str(ts)] = dict(population=int(pop.sum()), positives=int(yp.sum()),
                            auroc=None if len(np.unique(yp)) < 2 else float(roc_auc_score(yp, sp)),
                            auprc=None if len(np.unique(yp)) < 2 else float(average_precision_score(yp, sp)),
                            attenuation=None if not fd.any() else float(np.mean(1.0 - kept[fd])))
    return out


def cmd_engine(ds, force=False):
    name = f"engine_{ds}.json"
    record_path(name, force)                   # refuse before the engine runs, not after it
    D, G, Tr, sign = load_stream(ds)
    n_nodes = int(max(G.n, Tr.edges.max() + 1))
    out = dict(ds=ds, engine={k: engine_metrics(Tr, sign, n_nodes, v) for k, v in ENGINE_CONFIGS.items()},
               rolling=rolling(Tr, sign, n_nodes), provenance=C.provenance(D))
    save(name, out, force)


def cmd_trace(force=False):
    save("trace_symbolic.json", SC.verify_symbolic_example(), force)


def load_module(name):
    spec = importlib.util.find_spec(name)       # a fresh module object on every call, found via the package
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scem_localgdte(scem_cls=None):
    """The user's archived LocalGDTE+SCEM class. Its SCEM is keeper.scem.scem.SCEM, the one engine:
    updated/scem.py with four in-place index_add_ made out-of-place (lines 1105-1106, 1112-1113;
    approved by the user 2026-10-17, merged 2026-10-27), so training can back-propagate. Its Task 3.3
    option witness_weights stays None here unless scem_cls is witness_weighted_scem(...), and its phi stays
    learned unless scem_cls is fixed_phi_scem(...) (Task 3.4, G5).
    scem_cls, if given, replaces that class in a fresh copy of the module (forward_check's original arm)."""
    mod = load_module("keeper.backbone.local_model_scem")
    if scem_cls is not None:
        mod.SCEM = scem_cls
    return mod.LocalGDTE


def witness_weighted_scem(weights):
    """keeper.scem.scem.SCEM with fixed [T, n] witness weights (upgrade M-B, Task 3.3): every forward call
    passes them, because the archived LocalGDTE calls its SCEM without them. scem_localgdte(
    witness_weighted_scem(w)) builds the M-B LocalGDTE, with w from keeper.scem.witness_cred.level_weights.
    forward_check(..., original_scem=witness_weighted_scem(ones)) shows that all-ones weights leave the forward
    pass unchanged (tests/test_scem_upgrades.py). weights may be a tensor or a numpy array; the class keeps a
    private copy, so later changes to the caller's array cannot alias into it. (torch.tensor would copy too, but
    warns when given a tensor.) It is configured_scem(witness_weights=weights)."""
    if weights is None:
        raise ValueError("witness_weighted_scem needs weights; configured_scem() is the module without them")
    return configured_scem(witness_weights=weights)


def fixed_phi_scem(phi_fixed):
    """keeper.scem.scem.SCEM with its clock rate fixed (upgrade G5, Task 3.4): phi = phi_fixed, a constant without
    gradient, and the forced-close age A = erlang_horizon_A(N, phi_fixed), 13.28 snapshots at N = 2 and
    phi_fixed = 0.5, so that records commit inside the 21 training snapshots (E2). The archived LocalGDTE builds its
    SCEM without that argument, so the class supplies it: scem_localgdte(fixed_phi_scem(0.5)) builds the E2 model.
    phi_fixed must lie in the learned range [phi_min, phi_max] ([0.125, 0.5] at the module defaults, so 0.5 is
    phi_max); the module raises otherwise. phi_fixed=None gives the module unchanged (tests/test_scem_phi_fixed.py).
    It is configured_scem(phi_fixed=phi_fixed)."""
    return configured_scem(phi_fixed=phi_fixed)


def configured_scem(witness_weights=None, phi_fixed=None, **scem_kwargs):
    """keeper.scem.scem.SCEM with the module options of the E2 models combined (Task 4.5): fixed [T, n] witness
    weights (upgrade M-B, passed on every forward call; a private copy, as in witness_weighted_scem), phi_fixed
    (upgrade G5) and any constructor keyword in scem_kwargs, such as N_stages=1 or hazard=False (the E2 ablations).
    These override what the archived LocalGDTE passes. With every option at its default the class computes what
    keeper.scem.scem.SCEM computes (tests/test_e2_driver.py). SCEM's constructor swallows unknown keywords, so a
    caller asserts the built module's settings (keeper.scem.e2_driver.scem_settings)."""
    import torch
    weights = None if witness_weights is None else torch.as_tensor(witness_weights).detach().clone()
    options = dict(scem_kwargs, phi_fixed=phi_fixed)

    class ConfiguredSCEM(SC.SCEM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, **options})

        def forward(self, *args, witness_weights=None, **kwargs):
            chosen = weights if witness_weights is None else witness_weights
            return super().forward(*args, witness_weights=chosen, **kwargs)
    return ConfiguredSCEM


def load_original_module(path):
    """A stand-alone module file, the user's unpatched scem.py, imported under its own name without
    touching the import path. forward_check's original arm uses its SCEM class."""
    loader = importlib.machinery.SourceFileLoader("scem_original", os.path.abspath(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def same_parameters(s0, s1):
    """Two state dicts with the same non-empty key set and equal tensors."""
    import torch
    return bool(s0) and set(s0) == set(s1) and all(torch.equal(s0[k].cpu(), s1[k].cpu()) for k in s0)


def forward_check(ds, seed, device, original_scem=None):
    """Forward pass of two SCEM classes in the archived LocalGDTE, from the same seed: are the initial
    parameters, the embeddings and the auxiliary loss identical?

    The patched arm uses keeper.scem.scem.SCEM, the one engine, which carries the four approved
    out-of-place index_add lines. The original arm uses original_scem, the SCEM class of the user's
    unpatched scem.py (tag pre-revision-2026-10-26, read with load_original_module). With None both arms
    are the engine and only aux_loss_patched is evidence (the GPU stored-value test uses it so); the
    equiv subcommand therefore requires --original.
    device: the device keeper.laundering.core runs on ("cuda" or "cpu"); a mismatch raises, so that a
    stored GPU record is never compared with a CPU run. Returns the keys of equiv_<ds>_s<seed>.json.
    """
    import torch
    import torch.utils.checkpoint  # noqa: F401  local_model.py uses it; the user's pipeline imports it
                                   # via training / laundering/frugal_ssm.py, but this check never trains
    if device != C.DEV.type:
        raise ValueError(f"forward_check(device={device!r}), but keeper.laundering.core runs on {C.DEV}")
    C.set_deterministic(True)
    C.require_gpu()
    D, G, Tr, _ = load_stream(ds)
    C.apply_memory_profile(D)
    out = {}
    pipeline_class = C.LocalGDTE
    try:
        for fixed, scem_cls in ((False, original_scem), (True, SC.SCEM)):
            C.LocalGDTE = scem_localgdte(scem_cls)
            m = C.make_model(D, Tr, seed, D.tt)
            z = C.embed(m, Tr, D.tt)
            out[fixed] = (z.detach().cpu(), float(m.scem.last_aux_loss), m.state_dict())
    finally:
        C.LocalGDTE = pipeline_class
    (z0, a0, s0), (z1, a1, s1) = out[False], out[True]
    same_init = same_parameters(s0, s1)
    return dict(
        ds=ds, seed=seed, identical_initial_parameters=bool(same_init),
        max_abs_diff_embedding=float((z0 - z1).abs().max()), embeddings_equal=bool(torch.equal(z0, z1)),
        aux_loss_original=a0, aux_loss_patched=a1, provenance=C.provenance(D))


def cmd_equiv(ds, seed, original, force=False):
    """Forward pass of the user's unpatched SCEM, read from the file `original`, and of the engine, from
    the same seed: identical outputs? provenance.argv records the file. An existing record is replaced
    only with force (--force)."""
    name = f"equiv_{ds}_s{seed}.json"
    record_path(name, force)                   # refuse before the model work, not after it
    original_scem = load_original_module(original).SCEM
    save(name, forward_check(ds, seed, C.DEV.type, original_scem=original_scem), force)


def cmd_model(ds, seed, force=False):
    """The user's LocalGDTE+SCEM (archive local_model.py), trained by the user's pipeline."""
    import torch
    name = f"model_{ds}_s{seed}.json"
    record_path(name, force)           # refuse before the training, not after it
    C.LocalGDTE = scem_localgdte()     # glue: build the SCEM variant
    C.set_deterministic(True)
    C.require_gpu()
    D, G, Tr, _ = load_stream(ds)
    C.apply_memory_profile(D)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = C.train(C.make_model(D, Tr, seed, D.tt), Tr, D.tt)      # default 150 epochs
    train_s = time.time() - t0
    train_peak = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    z = C.embed(m, Tr, D.tt)
    infer_s = time.time() - t1
    infer_peak = torch.cuda.max_memory_allocated() / 2 ** 20
    tm = C.test_metrics(m, z, D, G)
    rows, t1m, _ = C.task_masks(G, D.tt)
    e = G.edges[rows][t1m]
    y = S.class_ids(G.labels[rows][t1m])
    p_dis = 1.0 - C.p_trust(m, z, e[:, 0], e[:, 1]).cpu().numpy()
    bins = np.minimum((p_dis * 10).astype(int), 9)
    ece = float(sum(np.mean(bins == b) * abs(p_dis[bins == b].mean() - y[bins == b].mean())
                    for b in range(10) if (bins == b).any()))
    save(name, dict(
        ds=ds, seed=seed, ordinary_mcc=tm["t1"]["mcc"], ordinary_auroc=tm["t1"]["auc"],
        n_test=tm["t1"]["n"], ece10=ece, train_seconds=round(train_s, 1),
        train_peak_mib=train_peak, infer_peak_mib=infer_peak,
        latency_us_per_event=1e6 * infer_s / max(len(Tr.edges), 1),
        scem_horizon_A=float(getattr(m.scem, "horizon_A", float("nan"))), provenance=C.provenance(D)), force)


def cmd_aggregate(force=False):
    import glob
    agg = {}
    for p in sorted(glob.glob(os.path.join(OUT, "model_*_s*.json"))):
        d = json.load(open(p))
        agg.setdefault(d["ds"], []).append(d)
    summary = {}
    for ds, runs in agg.items():
        summary[ds] = {}
        for k in ("ordinary_mcc", "ordinary_auroc", "ece10", "train_peak_mib", "latency_us_per_event"):
            xs = [r[k] for r in runs if r.get(k) is not None]
            summary[ds][k] = dict(mean=float(np.mean(xs)) if xs else None,
                                  sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else None, n=len(xs))
    save("model_summary.json", summary, force)


def main():
    global OUT                         # --out replaces the records folder for this process
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["engine", "trace", "equiv", "model", "aggregate"])
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--original", help="equiv: the user's unpatched scem.py "
                    "(git show pre-revision-2026-10-26:thecode/scem.py > FILE)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing record (any subcommand; or set KEEPER_OVERWRITE=1)")
    ap.add_argument("--out", help="folder for the records (default results/scem/stage_b/); aggregate "
                                  "also reads its model records from there")
    a = ap.parse_args()
    if a.out:
        OUT = os.path.abspath(a.out)
    if a.cmd == "equiv":
        if not a.original:
            ap.error("equiv needs --original FILE, the user's unpatched scem.py; write it with "
                     "'git show pre-revision-2026-10-26:thecode/scem.py > FILE' "
                     "(tests/test_scem_equivalence.py makes the same comparison)")
        if not os.path.isfile(a.original):
            ap.error(f"--original {a.original}: no such file")
    try:
        if a.cmd == "engine":
            cmd_engine(a.ds, a.force)
        elif a.cmd == "trace":
            cmd_trace(a.force)
        elif a.cmd == "equiv":
            cmd_equiv(a.ds, a.seed, a.original, a.force)
        elif a.cmd == "model":
            cmd_model(a.ds, a.seed, a.force)
        else:
            cmd_aggregate(a.force)
    except FileExistsError as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
