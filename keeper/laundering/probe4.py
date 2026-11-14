"""Reputation-laundering probe implementing proposal §3, §5a, §6 and §7 in full.

One (dataset, seed) run. For each sampled target T (strict distrust-majority, at
most ``budget`` distrust in-edges), it builds the attack plans and measures them
under the unlearning operators:

  plan            channel        operators
  direct (full)   C1 / C2-edge   unlink, unlink_frozen_x, exact_retrain,
                                 neggrad_plus, influence_newton
  direct greedy   C2-edge        unlink                (cost curve, k < |F|)
  account         C2-account     unlink, exact_retrain
  indirect white  C3             unlink, exact_retrain
  indirect grey   C3             unlink, exact_retrain (plan built on a surrogate)
  control         trust erasure  unlink, exact_retrain (matched count; None if T
                                 has no trust in-edge)
  placebo         far edges      unlink, exact_retrain (retraining's null)

Full-cohort H1: exact retraining over EVERY cohort target, in batches whose
receptive fields are disjoint (§7). The batching is validated against the
per-target retrains of the sampled targets.

Sampling (amendment A3): the draw is stratified. The positive-history stratum
gets up to half the draw; the rest comes from targets without one. Cohort-level
means are post-stratified. The draw depends only on the dataset (rng 7), so it
is identical across seeds, and seeds are paired.

Operator hyperparameters come from the pre-registered calibration
(``calibrate_operators.py``); the run refuses to start without it.

Run (from the repo root, on a GPU):
  CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/ps/bin/python -m keeper.laundering.probe4 --ds bitcoinalpha --seed 0
Rerun check of one target of the same draw (written to --out, never over a stored record):
  ... -m keeper.laundering.probe4 --ds bitcoinalpha --seed 0 --only-target 3509 --no-full-cohort --out FILE
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch

import numpy as np  # noqa: E402

from keeper.laundering import attack as A  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402
from keeper.laundering import unlearn as U  # noqa: E402
from keeper.laundering.threshold import held_out, optimal_threshold  # noqa: E402

DRAW_SEED = 7
DIRECT_OPS_ALL = ("unlink", "unlink_frozen_x", "exact_retrain", "neggrad_plus",
                  "influence_newton", "head_newton")
CHEAP_OPS = ("unlink", "exact_retrain")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-targets", type=int, default=50)
    ap.add_argument("--n-probe", type=int, default=256)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--max-c3", type=int, default=0,
                    help="0 = every C3 candidate; >0 = uniform random subsample, logged")
    ap.add_argument("--no-full-cohort", action="store_true",
                    help="skip the batched full-cohort H1 retrains")
    ap.add_argument("--no-greybox", action="store_true")
    ap.add_argument("--calibration", default=None,
                    help="calibration JSON (default results/laundering/calibration_<ds>_s0.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing final result file (the .partial.jsonl resume file is "
                         "appended to, never refused)")
    ap.add_argument("--only-target", type=int, default=None,
                    help="rerun check: process only this target of the draw; the draw and everything "
                         "computed before the target loop are unchanged; needs --out")
    a = ap.parse_args()
    if a.only_target is not None and a.out is None:
        ap.error("--only-target needs --out: a one-target record must not take a full run's file name")
    return a


def load_calibration(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"no calibration at {path}: run laundering.calibrate_operators "
                         "first (operator hyperparameters are pre-registered)")
    cal = json.load(open(path))
    out = {"influence_feasible": cal["influence"].get("feasible", True)}
    for key in ("neggrad", "influence", "head_newton"):
        if key == "influence" and not out["influence_feasible"]:
            out[key] = None
            continue
        sel = (cal.get(key) or {}).get("selected")
        if sel is None:
            raise SystemExit(f"calibration {path}: no feasible {key} setting; "
                             "widen the grid and re-calibrate before probing")
        out[key] = sel["hp"]
    out["source"] = path
    return out


def stratified_draw(cohort, n_pos_of, n):
    rng = np.random.default_rng(DRAW_SEED)
    pos = np.array([v for v in cohort if n_pos_of[int(v)] >= 1], dtype=np.int64)
    nop = np.array([v for v in cohort if n_pos_of[int(v)] == 0], dtype=np.int64)
    k_pos = min(len(pos), (n + 1) // 2)
    k_nop = min(len(nop), n - k_pos)
    k_pos = min(len(pos), n - k_nop)            # refill if the no-history stratum is short
    pick = np.concatenate([rng.choice(pos, k_pos, replace=False) if k_pos else [],
                           rng.choice(nop, k_nop, replace=False) if k_nop else []])
    return np.sort(pick.astype(np.int64)), dict(pos=len(pos), nopos=len(nop))


def select_targets(draw, only_target=None):
    """The targets one run processes: the whole draw, or (--only-target) that one target of the draw."""
    if only_target is None:
        return draw
    if int(only_target) not in draw.tolist():
        raise SystemExit(f"--only-target {only_target} is not in the draw {draw.tolist()}")
    return np.array([int(only_target)], dtype=np.int64)


class Ctx:
    """Everything shared by all targets of one (dataset, seed) run."""

    def __init__(self, a):
        self.a = a
        C.set_deterministic(True)
        C.require_gpu()
        self.D = C.load(a.ds)
        self.mem = C.apply_memory_profile(self.D)
        self.G = C.full_graph(self.D)
        self.tt = self.D.tt
        self.Tr = C.train_split(self.G, self.tt)
        self.panel = C.Panel(self.Tr, n_probe=a.n_probe, seed=1234)
        self.active = np.unique(self.Tr.edges.ravel())
        self.cohort, self.neg, self.pos = C.pick_targets(self.Tr, max_neg=a.budget,
                                                         strict=True)
        self.n_pos_of = {int(v): len(self.pos.get(int(v), [])) for v in self.cohort}
        self.hops = len(self.D.cfg["layers"])
        cal_path = a.calibration or os.path.join(C.RES, f"calibration_{a.ds}_s0.json")
        self.hp = load_calibration(cal_path)
        # weight-channel damping: the influence operator's when the full HVP is
        # feasible, else the head operator's (head-scoped weight channel)
        self.wc_damping = (self.hp["influence"]["damping"] if self.hp["influence_feasible"]
                           else self.hp["head_newton"]["damping"])
        self.direct_ops = tuple(op for op in DIRECT_OPS_ALL
                                if op != "influence_newton" or self.hp["influence_feasible"])

    def forget_mask(self, rows):
        m = np.zeros(len(self.Tr.edges), bool)
        m[np.asarray(rows, dtype=np.int64)] = True
        return m


# The sweep shares each GPU between several jobs, so an operator can hit a CUDA
# out-of-memory error that is caused by its neighbours, not by itself (observed
# 2026-10-10: wiki-rfa s4 died in the U4 Hessian-vector products while two other
# wiki-rfa jobs held 16 GB each). Every operator deep-copies m0 or retrains from
# scratch with a fixed seed, so retrying after the memory is released recomputes
# the same result. After OOM_RETRIES failed attempts the error is raised as before.
OOM_RETRIES = int(os.environ.get("PROBE4_OOM_RETRIES", "20"))
OOM_WAIT_S = int(os.environ.get("PROBE4_OOM_WAIT_S", "180"))


def run_operator(cx, m0, op, mask):
    import gc
    import torch
    for attempt in range(OOM_RETRIES + 1):
        try:
            return _run_operator_once(cx, m0, op, mask)
        except torch.OutOfMemoryError:
            if attempt == OOM_RETRIES:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            print(f"    CUDA OOM in {op}; retry {attempt + 1}/{OOM_RETRIES} in {OOM_WAIT_S}s",
                  flush=True)
            time.sleep(OOM_WAIT_S)


def _run_operator_once(cx, m0, op, mask):
    a, D, G, Tr, tt = cx.a, cx.D, cx.G, cx.Tr, cx.tt
    if op == "unlink":
        return U.unlink(m0, G, Tr, mask, tt, recompute_features=True)
    if op == "unlink_frozen_x":
        return U.unlink(m0, G, Tr, mask, tt, recompute_features=False)
    if op == "exact_retrain":
        return U.exact_retrain(D, G, tt, Tr.rid[mask], a.seed, a.epochs, self_term=C.self_term_of(m0))
    if op == "neggrad_plus":
        return U.neggrad_plus(m0, G, Tr, mask, tt, **cx.hp["neggrad"])
    if op == "influence_newton":
        return U.influence_unlearn(m0, G, Tr, mask, tt, **cx.hp["influence"])
    if op == "head_newton":
        return U.head_newton(m0, G, Tr, mask, tt, **cx.hp["head_newton"])
    raise ValueError(op)


def measure(cx, m0, base, v, mask, ops, forget_ce=False):
    """Gain, decomposition, percentile, utility and diagnostics per operator."""
    out = {}
    if not mask.any():
        return {op: None for op in ops}
    for op in ops:
        t0 = time.time()
        u = run_operator(cx, m0, op, mask)
        z = u.embed(cx.tt)
        rep = float(cx.panel.reputation(u.model, z, [v])[0])
        so, to = cx.panel.decompose(u.model, z, np.array([v]))
        rep_all = cx.panel.reputation(u.model, z, cx.active)
        tm = C.test_metrics(u.model, z, cx.D, u.G)
        rec = dict(rep=rep, gain=rep - base["r0"][v],
                   d_src=float(so[0] - base["src"][v]), d_tgt=float(to[0] - base["tgt"][v]),
                   pct_after=M.platform_percentile(rep, rep_all),
                   t1_mcc=tm["t1"]["mcc"], t1_auc=tm["t1"]["auc"], t2_mcc=tm["t2"]["mcc"],
                   d_t1_mcc=(None if tm["t1"]["mcc"] is None or base["t1"]["mcc"] is None
                             else tm["t1"]["mcc"] - base["t1"]["mcc"]),
                   seconds=round(time.time() - t0, 2),
                   info={k: w for k, w in u.info.items()
                         if isinstance(w, (int, float, str, bool))})
        if forget_ce:
            rec["forget_ce_served"] = U.forget_ce_served(u, u.Tr, cx.Tr, mask, cx.tt)
            rec["forget_ce_fullgraph"] = U.forget_ce(u, cx.Tr, mask, cx.tt)
        out[op] = rec
    if forget_ce and out.get("exact_retrain"):
        ref = out["exact_retrain"]["forget_ce_served"]
        for rec in out.values():
            rec["forget_gap_to_exact"] = abs(rec["forget_ce_served"] - ref)
    return out


def target_record(cx, m0, base, sur, v):
    a, Tr, tt = cx.a, cx.Tr, cx.tt
    rows_f = np.asarray(cx.neg[v], dtype=np.int64)
    k = len(rows_f)
    npos = cx.n_pos_of[v]
    rng = np.random.default_rng(10_000 * a.seed + v)
    rec = dict(target=v, n_neg=k, n_pos=npos, stratum=M.stratum_of(npos),
               r0=base["r0"][v], pct_before=base["pct0"][v])

    # counting baselines (A2): the direct plan erases all k distrust in-edges
    rec["count_gain"] = dict(laplace=float(M.counting_gain(npos, k, k, "laplace")),
                             raw=float(M.counting_gain(npos, k, k, "raw")))

    # --- direct, full budget (C1 subject-side; same edges as C2 edge-level)
    p_dir = A.plan_direct_full(Tr, v)
    mask_dir = p_dir.mask(Tr)
    rec["direct"] = dict(cost_C1=p_dir.cost,
                         cost_C2=A.cost_of(Tr, rows_f, "C2-rater-edge"),
                         ops=measure(cx, m0, base, v, mask_dir, cx.direct_ops, forget_ce=True))

    # --- white-box scorer + weight channel on the victim
    sc = A.Scorer(m0, Tr, tt, cx.panel, v)
    wc = A.WeightChannel(m0, Tr, tt, cx.panel, v, damping=cx.wc_damping)
    rec["weight_channel_cg"] = wc.cg

    # --- direct greedy cost curve (k' < k), unlink only
    curve = []
    for kk in range(1, k):
        p = A.plan_direct_greedy(sc, Tr, v, kk, wc)
        curve.append(dict(k=kk, cost=p.cost, unlink_gain=sc.unlink_gain(p.mask(Tr)),
                          trace=p.trace))
    rec["direct_greedy_curve"] = curve

    # --- account-level (C2)
    p_acc = A.plan_account(sc, Tr, v, r_budget=len(A.account_candidates(Tr, v)))
    rec["account"] = dict(cost=p_acc.cost, trace=p_acc.trace,
                          ops=measure(cx, m0, base, v, p_acc.mask(Tr), CHEAP_OPS))

    # --- indirect C3, white-box
    p_c3w = A.plan_indirect(sc, Tr, v, k, wc, "whitebox", a.max_c3, rng)
    rec["c3_white"] = dict(cost=p_c3w.cost, n_candidates=p_c3w.n_candidates,
                           trace=p_c3w.trace,
                           ops=measure(cx, m0, base, v, p_c3w.mask(Tr), CHEAP_OPS))
    wc.close()

    # --- indirect C3, grey-box: plan on the surrogate, execute on the victim
    if sur is not None:
        sm, spanel = sur
        ssc = A.Scorer(sm, Tr, tt, spanel, v)
        swc = A.WeightChannel(sm, Tr, tt, spanel, v, damping=cx.wc_damping)
        p_c3g = A.plan_indirect(ssc, Tr, v, k, swc, "greybox", a.max_c3, rng)
        swc.close()
        rec["c3_grey"] = dict(cost=p_c3g.cost, trace=p_c3g.trace,
                              same_as_white=bool(np.array_equal(p_c3g.rids, p_c3w.rids)),
                              ops=measure(cx, m0, base, v, p_c3g.mask(Tr), CHEAP_OPS))

    # --- control: erase a MATCHED count of trust in-edges (undefined if none)
    tpos = np.asarray(cx.pos.get(v, []), dtype=np.int64)
    if len(tpos):
        kc = min(k, len(tpos))
        rows_c = np.sort(rng.choice(tpos, kc, replace=False))
        rec["control"] = dict(n_erased=int(kc),
                              ops=measure(cx, m0, base, v, cx.forget_mask(rows_c), CHEAP_OPS))
    else:
        rec["control"] = None

    # --- placebo: k random far edges (retraining's null distribution)
    rows_p = A.placebo_rows(Tr, v, k, rng)
    rec["placebo"] = (dict(n_erased=int(len(rows_p)),
                           ops=measure(cx, m0, base, v, cx.forget_mask(rows_p), CHEAP_OPS))
                      if len(rows_p) else None)
    return rec


def identical_retrain(cx, m0):
    """A retrain of m0 with the same seed, data, budget and G4 flag, for the same-shape noise floor."""
    return C.train(C.make_model(cx.D, cx.Tr, cx.a.seed, cx.tt, self_term=C.self_term_of(m0)),
                   cx.Tr, cx.tt, epochs=cx.a.epochs)


def full_cohort_h1(cx, m0, base):
    """Exact retraining for every cohort target, batched by disjoint receptive field."""
    masks = {int(v): cx.forget_mask(cx.neg[int(v)]) for v in cx.cohort}
    r_new, batches = {}, []
    t0 = time.time()
    for batch, u in U.batched_exact_retrain(cx.D, cx.G, cx.Tr, cx.tt, cx.cohort, masks,
                                            cx.a.seed, cx.a.epochs, cx.hops,
                                            self_term=C.self_term_of(m0)):
        z = u.embed(cx.tt)
        reps = cx.panel.reputation(u.model, z, np.array(batch))
        r_new.update({int(v): float(r) for v, r in zip(batch, reps)})
        batches.append(len(batch))
        print(f"  full-cohort batch {len(batches)} ({len(batch)} targets, "
              f"{time.time() - t0:.0f}s)", flush=True)
    vs = [int(v) for v in cx.cohort]
    strata = [M.stratum_of(cx.n_pos_of[v]) for v in vs]
    r0 = np.array([base["r0"][v] for v in vs])
    rn = np.array([r_new[v] for v in vs])
    out = dict(n_batches=len(batches), batch_sizes=batches, r_new=r_new,
               flip={}, flip_tau={})
    for h in M.STRATA:
        sel = np.array([h == "all" or s == h for s in strata])
        out["flip"][h] = M.flip_rates(r0[sel], rn[sel], 0.5)
        out["flip"][h]["wilson95_literal"] = M.wilson_ci(
            out["flip"][h]["n_crossed"], out["flip"][h]["n"])
        out["flip_tau"][h] = M.flip_rates(r0[sel], rn[sel], base["tau"])
    out["gain"] = M.by_stratum(list(rn - r0), strata)
    return out


def summarise(cx, recs, cohort_out, base):
    strata = [r["stratum"] for r in recs]
    sizes = cx.strata_sizes

    def col(path):
        vals = []
        for r in recs:
            x = r
            for p in path:
                x = None if x is None else x.get(p) if isinstance(x, dict) else None
            vals.append(x)
        return vals

    s = dict(arms={})
    for arm, ops in (("direct", cx.direct_ops), ("account", CHEAP_OPS), ("c3_white", CHEAP_OPS),
                     ("c3_grey", CHEAP_OPS), ("control", CHEAP_OPS), ("placebo", CHEAP_OPS)):
        s["arms"][arm] = {}
        for op in ops:
            g = col([arm, "ops", op, "gain"])
            s["arms"][arm][op] = dict(
                gain=M.by_stratum(g, strata),
                gain_post_stratified=M.post_stratified_mean(g, strata, sizes),
                d_t1_mcc=M.by_stratum(col([arm, "ops", op, "d_t1_mcc"]), strata),
                pct_after=M.by_stratum(col([arm, "ops", op, "pct_after"]), strata))
    for kind in ("laplace", "raw"):
        cg = col(["count_gain", kind])
        for op in ("exact_retrain", "unlink"):
            mg = col(["direct", "ops", op, "gain"])
            s.setdefault("amplification", {}).setdefault(kind, {})[op] = {
                h: M.amplification([x for x, t in zip(mg, strata) if h == "all" or t == h],
                                   [x for x, t in zip(cg, strata) if h == "all" or t == h])
                for h in M.STRATA}
    for knowledge in ("c3_white", "c3_grey"):
        for op in ("exact_retrain", "unlink"):
            ind = [0.0 if (r.get(knowledge) is not None and r[knowledge]["ops"].get(op) is None)
                   else (None if r.get(knowledge) is None else r[knowledge]["ops"][op]["gain"])
                   for r in recs]
            dg = col(["direct", "ops", op, "gain"])
            s.setdefault("indirect_share", {}).setdefault(knowledge, {})[op] = {
                h: M.indirect_share([x for x, t in zip(ind, strata) if h == "all" or t == h],
                                    [x for x, t in zip(dg, strata) if h == "all" or t == h])
                for h in M.STRATA}
    # H1 from the sample's per-target exact retrains: literal §6 flip rate,
    # post-stratified to the cohort (feasible on every dataset, incl. Epinions)
    r0s = [r["r0"] for r in recs]
    rns = [r["r0"] + r["direct"]["ops"]["exact_retrain"]["gain"] for r in recs]
    s["h1_sample_post_stratified"] = {
        "p0.5": M.post_stratified_flip(r0s, rns, strata, sizes, 0.5),
        "tau": M.post_stratified_flip(r0s, rns, strata, sizes, base["tau"])}
    s["h1_sample_conditional"] = M.flip_rates(r0s, rns, 0.5)
    ex = col(["direct", "ops", "exact_retrain", "gain"])
    ng = col(["direct", "ops", "neggrad_plus", "gain"])
    d = [None if a is None or b is None else a - b for a, b in zip(ex, ng)]
    s["h3_exact_minus_neggrad"] = M.by_stratum(d, strata)
    dd = [x for x in d if x is not None]
    if len(dd) >= 6 and any(abs(x) > 0 for x in dd):
        from scipy.stats import wilcoxon
        s["h3_wilcoxon_p"] = float(wilcoxon(dd).pvalue)
    pl = [x for x in col(["placebo", "ops", "exact_retrain", "gain"]) if x is not None]
    s["placebo_floor"] = dict(n=len(pl), sd=float(np.std(pl, ddof=1)) if len(pl) > 1 else None,
                              max_abs=float(np.max(np.abs(pl))) if pl else None)
    s["full_cohort"] = None if cohort_out is None else {
        k: cohort_out[k] for k in ("n_batches", "flip", "flip_tau", "gain")}
    if cohort_out is not None:
        pt = [(r["direct"]["ops"]["exact_retrain"]["gain"],
               cohort_out["r_new"][r["target"]] - r["r0"]) for r in recs]
        a_, b_ = np.array(pt).T
        s["batching_validation"] = dict(
            n=len(pt), corr=float(np.corrcoef(a_, b_)[0, 1]) if len(pt) > 2 else None,
            mean_abs_diff=float(np.mean(np.abs(a_ - b_))),
            max_abs_diff=float(np.max(np.abs(a_ - b_))))
    return s


def main():
    a = parse()
    T0 = time.time()
    out_name = a.out or f"probe4_{a.ds}_s{a.seed}.json"
    C.result_path(out_name, a.force)          # refuse an existing final record before the work, not after it
    cx = Ctx(a)
    tg, cx.strata_sizes = stratified_draw(cx.cohort, cx.n_pos_of, a.n_targets)
    tg = select_targets(tg, a.only_target)
    partial = (out_name if os.path.isabs(out_name) else os.path.join(C.RES, out_name)) \
        .replace(".json", ".partial.jsonl")

    m0 = C.train(C.make_model(cx.D, cx.Tr, a.seed, cx.tt), cx.Tr, cx.tt, epochs=a.epochs)
    z0 = C.embed(m0, cx.Tr, cx.tt)
    base_tm = C.test_metrics(m0, z0, cx.D, cx.G)
    e_h, y_h = held_out(cx.G, cx.D)
    tau = optimal_threshold(C.p_trust(m0, z0, e_h[:, 0], e_h[:, 1]).cpu().numpy(), y_h)
    r0c = cx.panel.reputation(m0, z0, cx.cohort)
    so, to = cx.panel.decompose(m0, z0, cx.cohort)
    rep_all0 = cx.panel.reputation(m0, z0, cx.active)
    base = dict(r0={int(v): float(r) for v, r in zip(cx.cohort, r0c)},
                src={int(v): float(x) for v, x in zip(cx.cohort, so)},
                tgt={int(v): float(x) for v, x in zip(cx.cohort, to)},
                pct0={int(v): M.platform_percentile(r, rep_all0) for v, r in zip(cx.cohort, r0c)},
                t1=base_tm["t1"], t2=base_tm["t2"], tau=float(tau))
    # same-shape noise floor: an identical retrain, queried with the SAME batch
    m_rep = identical_retrain(cx, m0)
    floor_same = float(np.abs(cx.panel.reputation(m_rep, C.embed(m_rep, cx.Tr, cx.tt),
                                                  cx.cohort) - r0c).max())
    del m_rep
    print(f"[{a.ds} s{a.seed}] train={len(cx.Tr.edges):,} T1 MCC={base_tm['t1']['mcc']:.4f} "
          f"T2 MCC={base_tm['t2']['mcc']} | cohort {len(cx.cohort)} "
          f"(pos {cx.strata_sizes['pos']}, nopos {cx.strata_sizes['nopos']}) | "
          f"sample {len(tg)} | same-shape floor {floor_same:.2e} | "
          f"hp {cx.hp['neggrad']} {cx.hp['influence']}", flush=True)

    sur = None if a.no_greybox else A.surrogate(cx.D, cx.Tr, cx.tt, a.seed, a.epochs,
                                                a.n_probe)
    done = {}
    if os.path.exists(partial):
        for line in open(partial):
            r = json.loads(line)
            if r.get("_code") == C.CODE_SHA256_AT_IMPORT:
                done[r["target"]] = r
        print(f"  resuming: {len(done)} targets already done", flush=True)
    recs = []
    for i, v in enumerate(tg.tolist()):
        if v in done:
            recs.append(done[v])
            continue
        t1 = time.time()
        r = target_record(cx, m0, base, sur, v)
        r["_code"] = C.CODE_SHA256_AT_IMPORT
        r["seconds"] = round(time.time() - t1, 1)
        with open(partial, "a") as f:
            f.write(json.dumps(C._finite(r), default=C._json_default) + "\n")
        recs.append(r)
        d = r["direct"]["ops"]
        print(f"  target {i + 1}/{len(tg)} v={v} {r['stratum']} k={r['n_neg']} "
              + " ".join(f"{op} {d[op]['gain']:+.4f}" for op in cx.direct_ops if d.get(op))
              + f" ({r['seconds']:.0f}s)", flush=True)

    cohort_out = None if a.no_full_cohort else full_cohort_h1(cx, m0, base)
    out = dict(ds=a.ds, seed=a.seed, epochs=a.epochs, budget=a.budget,
               n_probe=len(cx.panel.sources), hops=cx.hops, memory_profile=cx.mem,
               calibration=cx.hp, cohort_size=int(len(cx.cohort)),
               strata_sizes=cx.strata_sizes, sample=tg.tolist(), draw_seed=DRAW_SEED,
               base=dict(t1=base["t1"], t2=base["t2"], tau=base["tau"],
                         r0_cohort_mean=float(np.mean(r0c))),
               noise_floor_same_shape=floor_same,
               summary=summarise(cx, recs, cohort_out, base),
               full_cohort=cohort_out, targets=recs,
               runtime_s=round(time.time() - T0, 1))
    if a.only_target is not None:
        out["only_target"] = int(a.only_target)    # a one-target rerun check, not a full run
    path = C.dump(out, out_name, cx.D, force=a.force)
    print(f"wrote {path} ({out['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
