"""E9a: the self-term ablation of the LocalGDTE backbone (protocol Section 2.11, hypothesis H8), with E8's dumps.

One GPU run per (network, seed, self_term), from the repository root of a worktree fixed at the run commit:
  python -m keeper.laundering.e9a_runner --ds DS --seed S --self-term {off,on} --epochs 150 \
      --prereg docs/preregistration.md --out ABS.json --dump-dir ABS_DIR

Context: probe4's own (keeper.laundering.probe4.Ctx: dataset, full graph, training split, the frozen panel of 256
sources, the active accounts, the distrust and trust in-edges). Ctx also reads probe4's operator calibration file,
which E9a does not use. The base model is LocalGDTE, ungated, built by core.make_model(..., self_term=...) and
trained by core.train for --epochs epochs.

Targets, in this order (target_lists):
  drawn  probe4's 50 drawn targets: probe4.stratified_draw over its cohort (strict distrust majority, at most 3
         distrust in-edges; rng 7, so the draw depends on the dataset only and is the same in every seed);
  ge4    the separate stratum: every account with a strict distrust majority and at least 4 distrust in-edges.
U1 is probe4's unlink operator (probe4.run_operator(..., "unlink", mask)): the mask holds every distrust in-edge of the
target, the parameters are kept and the features are recomputed from the cut graph; nothing is retrained. Per target:
r0 and r, the target's proxy before and after U1; gain = r - r0; pct_after = metrics.platform_percentile(r, every
active account's proxy on the post-U1 model), probe4's definition; n_after, its live incoming training ratings;
edgeless = (n_after == 0). Every proxy comes from one panel query over all active accounts (the base model's, and
each post-U1 model's), so r is the target's own entry of its population row, as E8 requires, and r0 and r come from
queries of the same shape.

Dump (keeper.analysis.e9a_e8 reads it): <dump-dir>/<result stem>.npz with
  active_ids [A], base_proxy [A], n_in [n] (incoming training ratings per account id),
  test_rows, test_dst, test_y, test_p: every row of the test snapshot in cold_start.held_out_snapshot order, with
      its destination, class id (1 = distrust) and the base model's P(trust),
  target_ids [T], target_n [T] (n_after), population [T, A] (row i: every active account's proxy after target i's U1).
Proxies and probabilities are float32 values (the panel averages float32 probabilities), stored as float32 when
that is lossless and as float64 otherwise. The result JSON (core.dump, atomic, R27) records the dump's sha256. The
dump is written first, replacing an orphan left by a failed attempt: the result file is the run's done mark, and
core.result_path refuses an existing one before any work.
"""
import argparse
import os
import sys
import time
import warnings
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch, as probe4

import numpy as np  # noqa: E402

from keeper.analysis import cold_start as CS  # noqa: E402
from keeper.analysis.provenance import git_state  # noqa: E402
from keeper.cppr.protocol import preregistration, sha256_file  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402
from keeper.laundering import probe4 as P4  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402

N_DRAWN = 50            # probe4's --n-targets default
BUDGET = 3              # probe4's --budget default: its cohort's cap on distrust in-edges
N_PROBE = 256           # probe4's --n-probe default
STRATUM_MIN_NEG = 4     # the separate stratum: at least this many distrust in-edges
GROUPS = ("drawn", "ge4")
SELF_TERMS = ("off", "on")


def target_lists(Tr) -> tuple:
    """(drawn, stratum, strata_sizes) on the training graph Tr: probe4's 50 drawn targets with its stratum sizes, and
    every strict distrust-majority account with at least STRATUM_MIN_NEG distrust in-edges (sorted ids)."""
    cohort, neg, pos = C.pick_targets(Tr, max_neg=BUDGET, strict=True)
    drawn, sizes = P4.stratified_draw(cohort, {int(v): len(pos.get(int(v), [])) for v in cohort}, N_DRAWN)
    majority, _, _ = C.pick_targets(Tr, max_neg=len(Tr.edges), strict=True)
    stratum = np.array([v for v in majority.tolist() if len(neg[v]) >= STRATUM_MIN_NEG], dtype=np.int64)
    return drawn, stratum, sizes


def u1(cx, m0, v: int, base_all: np.ndarray, j: int) -> tuple:
    """(record, population row) of target v at position j of cx.active: U1 over all of v's distrust in-edges."""
    rows = np.asarray(cx.neg[v], dtype=np.int64)
    u = P4.run_operator(cx, m0, "unlink", cx.forget_mask(rows))
    into = u.Tr.edges[:, 1] == v
    if (into & S.neg_mask(u.Tr.labels)).any():
        raise RuntimeError(f"U1 left a distrust in-edge of target {v}: the mask must cover all of them")
    post = cx.panel.reputation(u.model, u.embed(cx.tt), cx.active)
    r, r0 = float(post[j]), float(base_all[j])
    n_after = int(into.sum())
    rec = dict(target=v, n_neg=len(rows), n_pos=len(cx.pos.get(v, [])), r0=r0, r=r, gain=r - r0,
               pct_before=M.platform_percentile(r0, base_all), pct_after=M.platform_percentile(r, post),
               n_after=n_after, edgeless=n_after == 0)
    return rec, post


def _compact(x: np.ndarray) -> np.ndarray:
    """x as float32 when the cast loses nothing, else unchanged."""
    x32 = x.astype(np.float32)
    return x32 if np.array_equal(x32, x) else x


def dump_arrays(cx, base_all, test_p, rows, dst, y, targets, recs, population) -> dict:
    return dict(active_ids=np.asarray(cx.active, dtype=np.int64), base_proxy=_compact(base_all),
                n_in=CS.incoming_counts(cx.Tr.edges[:, 1], cx.D.n), test_rows=rows.astype(np.int64),
                test_dst=dst.astype(np.int64), test_y=y.astype(np.int64), test_p=_compact(test_p),
                target_ids=targets.astype(np.int64), target_n=np.array([r["n_after"] for r in recs], dtype=np.int64),
                population=_compact(population))


def _gpu_peak_mib():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None


def _work(a) -> tuple:
    """The model work of a run: (cx, config, base test metrics, per-target records, dump arrays)."""
    cx = P4.Ctx(SimpleNamespace(ds=a.ds, n_probe=N_PROBE, budget=BUDGET, calibration=None, seed=a.seed,
                                epochs=a.epochs))
    drawn, stratum, sizes = target_lists(cx.Tr)
    if a.smoke:
        drawn, stratum = drawn[:a.smoke], stratum[:a.smoke]
    targets = np.concatenate([drawn, stratum]).astype(np.int64)
    m0 = C.train(C.make_model(cx.D, cx.Tr, a.seed, cx.tt, self_term=a.self_term == "on"), cx.Tr, cx.tt,
                 epochs=a.epochs)
    z0 = C.embed(m0, cx.Tr, cx.tt)
    base_all = cx.panel.reputation(m0, z0, cx.active)
    base_tm = C.test_metrics(m0, z0, cx.D, cx.G)
    rows, dst, y = CS.held_out_snapshot(cx.G, cx.tt)
    e = cx.G.edges[rows]
    test_p = C.p_trust(m0, z0, e[:, 0], e[:, 1]).cpu().numpy() if len(rows) else np.zeros(0, np.float32)
    position = {v: j for j, v in enumerate(cx.active.tolist())}
    recs, population, t0 = [], np.empty((len(targets), len(cx.active))), time.time()
    for i, v in enumerate(targets.tolist()):
        rec, population[i] = u1(cx, m0, v, base_all, position[v])
        recs.append(dict(rec, group=GROUPS[0] if i < len(drawn) else GROUPS[1]))
        if (i + 1) % 50 == 0 or i + 1 == len(targets):
            print(f"  U1 {i + 1}/{len(targets)} targets ({time.time() - t0:.0f}s)", flush=True)
    config = dict(ds=a.ds, seed=a.seed, self_term=a.self_term == "on", epochs=a.epochs, model="LocalGDTE ungated",
                  operator="unlink", recompute_features=True, n_probe=int(len(cx.panel.sources)), budget=BUDGET,
                  draw_seed=P4.DRAW_SEED, n_drawn=N_DRAWN, stratum_min_neg=STRATUM_MIN_NEG,
                  targets_drawn=drawn.tolist(), targets_stratum=stratum.tolist(), strata_sizes=sizes,
                  n_active=int(len(cx.active)), smoke=a.smoke)
    return cx, config, base_tm, recs, dump_arrays(cx, base_all, test_p, rows, dst, y, targets, recs, population)


def run(a) -> dict:
    """One E9a run; writes the dump and then the result JSON, and returns the result."""
    T0 = time.time()
    if not a.out.endswith(".json"):
        raise ValueError(f"--out must name a .json file, got {a.out}")
    out_path = C.result_path(a.out, a.force)                     # refuse an existing result before the training
    dump_path = os.path.join(os.path.abspath(a.dump_dir), os.path.basename(out_path)[:-len(".json")] + ".npz")
    prereg = preregistration(a.prereg)
    for folder in (os.path.dirname(out_path), os.path.dirname(dump_path)):
        os.makedirs(folder, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cx, config, base_tm, recs, arrays = _work(a)
    C.write_atomic(dump_path, lambda f: np.savez(f, **arrays), replace=True, binary=True)
    seen = sorted({f"{w.category.__name__}: {str(w.message)[:300]}" for w in caught})
    out = dict(experiment="E9a", ds=a.ds, seed=a.seed, self_term=config["self_term"], epochs=a.epochs,
               config=config, preregistration=prereg, memory_profile=cx.mem, base=dict(test_metrics=base_tm),
               targets=recs,
               dump=dict(path=dump_path, sha256=sha256_file(dump_path), n_bytes=os.path.getsize(dump_path),
                         arrays={k: dict(dtype=str(v.dtype), shape=list(v.shape)) for k, v in arrays.items()}),
               warnings=seen, gpu_peak_mib=_gpu_peak_mib(), runtime_s=round(time.time() - T0, 1))
    C.dump(dict(out, provenance=dict(C.provenance(cx.D), **git_state())), out_path, cx.D, force=a.force)
    drawn = [r["gain"] for r in recs if r["group"] == GROUPS[0]]
    for line in seen:
        print(f"  warning: {line}")
    print(f"[E9a {a.ds} s{a.seed} self_term={a.self_term}] T1 MCC={base_tm['t1']['mcc']} | {len(recs)} targets | "
          f"drawn mean U1 gain {np.mean(drawn):+.4f} | {len(seen)} distinct warnings | "
          f"dump {out['dump']['n_bytes'] / 2 ** 20:.1f} MiB | {out['runtime_s']}s")
    print(f"wrote {out_path}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E9a: one (network, seed, self_term) run of the self-term ablation.")
    ap.add_argument("--ds", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--self-term", choices=SELF_TERMS, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--prereg", required=True, metavar="PATH", help="the protocol file; its sha256 is recorded")
    ap.add_argument("--out", required=True, help="the result JSON (absolute path, or a name under results/laundering)")
    ap.add_argument("--dump-dir", required=True, help="the folder of the .npz dump (git-ignored results/dumps/e9a)")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="test only: the first N targets of each group (the analysis refuses such a run)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing result (or set KEEPER_OVERWRITE=1)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.smoke is not None and a.smoke < 1:
        raise SystemExit("--smoke must be at least 1")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
