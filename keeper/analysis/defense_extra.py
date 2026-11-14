"""Defense re-analysis of the stored results: legacy tier coverage, single-warning revocation and the
pre-revision CPPR seed statistics (Task 2.2).

No model is trained or run.
1. Tier coverage. Each CPPR network's training graph is rebuilt with the dataset loader of
   keeper.laundering.core (data processing only, CPU). The legacy tiers come from
   keeper.credibility.tiers.legacy_bfs, and the endorsed set D from a port of the archival review script
   docs/reviews/2026-10-26/scripts/review_tiers.py, which stays unchanged. The port is checked against
   legacy_bfs on every network.
2. Single-warning revocation. The revoked fraction of an escrowed rating whose sender receives one warning
   comes from the closed forms of keeper.scem.scem, under two conventions. The paper's Eq. 8
   (revocation_curve_C) keeps the hazard on until the record has released everything. The engine's rules
   end the hazard when the witness evidence expires, H snapshots after it matures, or when the record
   closes at age A. The hazard that one witness produces follows the engine's hazard formula. The engine
   constants are read from the defaults of keeper.scem.scem.run_scem_streaming_weights_np.
3. CPPR seed statistics. The seed-level paired statistics come from the stored CPPR runs
   results/laundering/cpgate_count_<ds>_s<seed>.json. They are pre-revision runs, superseded by E3.

Run from the repository root, CPU only:
  CUDA_VISIBLE_DEVICES="" python -m keeper.analysis.defense_extra [--force]
It writes results/analysis/defense_extra.json and results/analysis/defense_extra.md.
"""
import argparse
import datetime
import inspect
import json
import math
import os
import platform
import socket
import sys

import numpy as np
from scipy.stats import t as student_t

from keeper.analysis.bootstrap import N_BOOT
from keeper.analysis.intervals import BOOT_SEED, Block, Boot
from keeper.analysis.laundering_extra import Inputs, sha256_file
from keeper.analysis.provenance import git_state
from keeper.credibility.tiers import legacy_bfs
from keeper.laundering import core as C
from keeper.laundering import signed as S
from keeper.scem import scem as SC

__all__ = ["ENGINE", "Inputs", "endorsed_set", "first_snapshots", "tier_coverage", "horizon_A",
           "revoked_no_expiry", "revoked_engine_rules", "revoked_by_routing", "witness_hazard",
           "witnesses_to_reach", "revocation_grid", "hazard_map", "paired_t", "check_cppr_runs",
           "load_cppr_runs", "cppr_seed_block", "analyse", "main"]

NETWORKS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")      # the three CPPR networks
SEEDS = tuple(range(5))
SEED_LAST_SNAPSHOT = 4            # legacy seeds: majority-positive in-edges in snapshots 0 to 4
N_TIERS = 4
CENSUS = "census of the training graph (deterministic, no interval)"
ANALYSIS = os.path.join(C.ROOT, "results", "analysis")
CPPR_FILE = "results/laundering/cpgate_count_{ds}_s{seed}.json"
CPPR_GATE = "count"               # CPPR's gate of Eq. (15); the per-snapshot gate is the ablation
PRE_REVISION = "pre-revision runs (superseded by E3)"
LAMBDAS = (0.10, 0.53, 1.0, 2.0)  # the review's hazards (0.53 and 0.10 from one witness) and the ceiling 2
LAGS = (0, 1, 3)                  # snapshots from the rating to the warning
KAPPAS = ((1.0, "fully reliable witness"), (0.35, "reliability floor up to snapshot 4"),
          (0.20, "reliability floor after snapshot 4"))
CEILING_TARGET = 1.9              # the supplement: reaching 1.9 takes at least seven fully reliable witnesses
MAX_WITNESSES = 100
MODULE_FILES = ("keeper/analysis/defense_extra.py", "keeper/analysis/defense_extra_md.py",
                "keeper/analysis/bootstrap.py", "keeper/analysis/intervals.py",
                "keeper/analysis/laundering_extra.py", "keeper/analysis/laundering_extra_md.py",
                "keeper/credibility/tiers.py", "keeper/scem/scem.py", "keeper/analysis/provenance.py")


def _engine_defaults() -> dict:
    params = inspect.signature(SC.run_scem_streaming_weights_np).parameters
    return {name: p.default for name, p in params.items() if p.default is not inspect.Parameter.empty}


_DEFAULTS = _engine_defaults()
ENGINE = dict(
    N=_DEFAULTS["N"], phi=_DEFAULTS["phi"], L_f=_DEFAULTS["L_f"], H=_DEFAULTS["H"], q=_DEFAULTS["q_quantile"],
    lambda_max=_DEFAULTS["lambda_max"], gamma=tuple(_DEFAULTS["gamma_sigma"]), L_h=_DEFAULTS["L_h"],
    cap_c=_DEFAULTS["cap_c"],
    # Fixed in the engine's body, not parameters; tests/test_defense_extra.py runs the engine to pin them.
    psi={"positive": (0.80, 0.20), "negative": (0.20, 0.80)},
    kappa_floor={"t<=4": 0.35, "t>4": 0.20})


# ------------------------------------------------------------------------ 1. legacy tier coverage
def endorsed_set(Tr, n_nodes: int) -> tuple:
    """(seeds, D): the seeded endorsement pass of legacy_levels, replicated as review_tiers.py does.

    Seeds have more positive than negative training in-edges in snapshots 0 to 4. One pass over the edges
    in their stored order then adds the target of every positive rating whose sender is already in D.
    """
    y_neg = S.class_ids(Tr.labels)
    early_pos = np.zeros(n_nodes, dtype=np.int64)
    early_neg = np.zeros(n_nodes, dtype=np.int64)
    for (u, v), is_neg, sn in zip(Tr.edges, y_neg, Tr.snap):
        if int(sn) <= SEED_LAST_SNAPSHOT:
            if is_neg:
                early_neg[int(v)] += 1
            else:
                early_pos[int(v)] += 1
    seeds = set(np.where(early_pos > early_neg)[0].tolist())
    members = set(seeds)
    for (u, v), is_neg in zip(Tr.edges, y_neg):
        if not is_neg and int(u) in members:
            members.add(int(v))
    return seeds, members


def first_snapshots(Tr, n_nodes: int) -> np.ndarray:
    """Each node's first training snapshot as either endpoint of an edge; -1 for a node without one."""
    edges = np.asarray(Tr.edges, dtype=np.int64)
    snap = np.asarray(Tr.snap, dtype=np.int64)
    never = np.iinfo(np.int64).max
    first = np.full(n_nodes, never)
    np.minimum.at(first, edges[:, 0], snap)
    np.minimum.at(first, edges[:, 1], snap)
    first[first == never] = -1
    return first


def tier_coverage(Tr, n_nodes: int) -> dict:
    """Seeds, endorsed set D, late joiners and the tier mix of the active accounts under legacy_bfs.

    Raises AssertionError if the replicated D disagrees with legacy_bfs: Tier >= 1 must be exactly the
    accounts whose positive in-edges from members of D outnumber their negative in-edges.
    """
    levels = legacy_bfs(Tr, n_nodes)
    seeds, members = endorsed_set(Tr, n_nodes)
    edges = np.asarray(Tr.edges, dtype=np.int64)
    is_neg = S.neg_mask(Tr.labels)
    in_D = np.zeros(n_nodes, dtype=bool)
    in_D[sorted(members)] = True
    endorsing = ~is_neg & in_D[edges[:, 0]]                       # positive ratings sent by members of D
    pos_from_D = np.bincount(edges[endorsing, 1], minlength=n_nodes)
    neg_in = np.bincount(edges[is_neg, 1], minlength=n_nodes)
    if not np.array_equal(pos_from_D > neg_in, levels > 0):
        raise AssertionError("the replicated endorsed set D disagrees with legacy_bfs: Tier >= 1 must be "
                             "the accounts whose positive in-edges from D outnumber their negative ones")
    active = np.unique(edges.reshape(-1))
    late = active[first_snapshots(Tr, n_nodes)[active] > SEED_LAST_SNAPSHOT]
    counts = np.bincount(levels[active], minlength=N_TIERS)
    n_active = int(len(active))
    return dict(n_nodes=int(n_nodes), n_active=n_active, n_seeds=len(seeds), n_D=len(members),
                D_share_of_active=len(members) / n_active,
                n_D_positive_raters=int(len(np.unique(edges[endorsing, 0]))),
                n_late=int(len(late)), n_late_in_D=int(in_D[late].sum()),
                late_in_D_share=float(in_D[late].mean()) if len(late) else None,
                tier_counts=counts.tolist(), tier_shares=(counts / n_active).tolist(),
                tier0_share=float(counts[0] / n_active), n_endorsed=int((levels > 0).sum()))


def coverage_block(cov: dict, boot: Boot) -> dict:
    """tier_coverage as an output block: every scalar is a census (ci95 null), lists go to extra."""
    b = Block("tier_coverage", boot)
    for field, value in cov.items():
        if isinstance(value, list):
            b.extra[field] = value
        else:
            b.exact(field, value, CENSUS)
    return b.to_dict()


# ------------------------------------------------------------------------ 2. single-warning revocation
def horizon_A() -> float:
    """The engine's forced-close age A = F^-1_Erlang(N, phi)(q)."""
    return float(SC.erlang_horizon_A(ENGINE["N"], ENGINE["phi"], q=ENGINE["q"]))


def revoked_no_expiry(lag: float, lam: float) -> float:
    """Eq. 8 (revocation_curve_C): revoked fraction under a constant hazard lam from age lag + L_f on,
    with neither expiry of the warning nor closure of the record (the paper's curve and census)."""
    return float(SC.revocation_curve_C(lag, ENGINE["N"], ENGINE["phi"], ENGINE["L_f"], lam)[1])


def revoked_engine_rules(lag: float, lam: float, H: float | None = None, A: float | None = None) -> float:
    """Revoked fraction under the engine's rules, for a warning issued ``lag`` snapshots after the rating.

    The warning matures at age x0 = lag + L_f, and its evidence is live for H snapshots. The record closes
    at age A, committing what is left. So the hazard lam acts on [x0, x1), x1 = min(x0 + H, A). The mass
    released before x0, P(N, phi x0), is frozen. During [x0, x1) the open mass is thinned while it keeps
    releasing (segment_release_increment_np, Eq. 2). The mass still open at x1, exp(-lam (x1 - x0))
    Q(N, phi x1), is committed at closure or released later without a hazard. A record whose warning
    matures at or after A has closed and loses nothing. H and A default to the engine's values.
    """
    if lag < 0:
        raise ValueError(f"lag must be at least 0 snapshots, got {lag}")
    N, phi = ENGINE["N"], ENGINE["phi"]
    H = ENGINE["H"] if H is None else H
    A = horizon_A() if A is None else A
    x0 = lag + ENGINE["L_f"]
    if x0 >= A:
        return 0.0
    x1 = min(x0 + H, A)
    _, released_before = SC.erlang_Q_P_np(N, phi * x0)
    released_during = SC.segment_release_increment_np(N, phi, lam, x0, x1, 0.0)
    still_open, _ = SC.erlang_Q_P_np(N, phi * x1)
    committed = float(released_before) + float(released_during) + math.exp(-lam * (x1 - x0)) * float(still_open)
    return 1.0 - committed


def revoked_by_routing(lag: float, lams, psi) -> float:
    """A record's revoked fraction in the engine: its per-channel fractions mixed by its routing psi."""
    return float(sum(p * revoked_engine_rules(lag, lam) for p, lam in zip(psi, lams)))


def witness_hazard(kappas) -> dict:
    """The engine's sender hazard from distinct witnesses with reliabilities ``kappas``.

    Each witness enters xi^{V1,-} with its reliability, capped at c (the per-contributor l1 cap);
    h = L_h xi, nu = softplus(h) - log 2 (floored at 0) and lam^sigma = lambda_max (1 - exp(-gamma_sigma
    nu / lambda_max)) for each channel sigma, exactly as run_scem_streaming_weights_np computes them.
    """
    xi = sum(min(float(k), ENGINE["cap_c"]) for k in kappas)
    h = ENGINE["L_h"] * max(xi, 0.0)
    nu = max(math.log1p(math.exp(min(h, 40.0))) - math.log(2.0), 0.0)
    lam_max = ENGINE["lambda_max"]
    lam = [lam_max * (1.0 - math.exp(-(g * nu) / lam_max)) for g in ENGINE["gamma"]]
    return dict(xi=xi, h=h, nu=nu, lam=lam)


def witnesses_to_reach(target: float, kappa: float = 1.0, max_witnesses: int = MAX_WITNESSES) -> int | None:
    """Fewest distinct witnesses at reliability kappa whose channel-1 hazard reaches target; None if
    max_witnesses do not (the hazard stays below lambda_max)."""
    for k in range(1, max_witnesses + 1):
        if witness_hazard([kappa] * k)["lam"][0] >= target:
            return k
    return None


def revocation_grid(lambdas=LAMBDAS, lags=LAGS) -> list:
    """One row per (lam, lag): the revoked fraction under the engine's rules and without expiry."""
    rows = []
    for lam in lambdas:
        for lag in lags:
            r = revoked_engine_rules(lag, lam)
            rows.append(dict(lam=lam, lag=lag, revoked_engine_rules=r, durable_engine_rules=1.0 - r,
                             revoked_no_expiry=revoked_no_expiry(lag, lam)))
    return rows


def hazard_map(kappas=KAPPAS, lags=LAGS) -> list:
    """One row per witness reliability: the hazard of one warning and what it revokes at each lag."""
    rows = []
    for kappa, label in kappas:
        hz = witness_hazard([kappa])
        cells = []
        for lag in lags:
            channel = [revoked_engine_rules(lag, lam) for lam in hz["lam"]]
            no_expiry = [revoked_no_expiry(lag, lam) for lam in hz["lam"]]
            cell = dict(lag=lag, channel=channel, channel_no_expiry=no_expiry)
            for routing, psi in ENGINE["psi"].items():
                cell[f"{routing}_rating"] = revoked_by_routing(lag, hz["lam"], psi)
                cell[f"{routing}_rating_no_expiry"] = float(sum(p * r for p, r in zip(psi, no_expiry)))
            cells.append(cell)
        rows.append(dict(kappa=kappa, label=label, **hz, revoked=cells))
    return rows


def single_warning_block() -> dict:
    n = witnesses_to_reach(CEILING_TARGET)
    shown = (n or MAX_WITNESSES) + 1
    return dict(engine=dict(ENGINE, A=horizon_A()), grid=revocation_grid(), hazard_map=hazard_map(),
                witnesses_to_reach=dict(target=CEILING_TARGET, kappa=1.0, n=n,
                                        lam1_by_count=[witness_hazard([1.0] * k)["lam"][0]
                                                       for k in range(1, shown + 1)]))


# ------------------------------------------------------------------------ 3. CPPR seed-level statistics
def paired_t(a, b) -> dict:
    """Paired t statistic of a - b over seeds (df = n - 1); t and p are None if every difference is equal."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = int(len(d))
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0.0:
        t = p = None
    else:
        t = mean / (sd / math.sqrt(n))
        p = float(2.0 * student_t.sf(abs(t), n - 1))
    return dict(n=n, df=n - 1, mean=mean, sd=sd, t=t, p_two_sided=p)


def check_cppr_runs(ds: str, runs: list) -> None:
    """Refuse runs that are not the count-gate runs of (ds, seed 0..4), in seed order."""
    if len(runs) != len(SEEDS):
        raise ValueError(f"{ds}: {len(runs)} cpgate runs, expected {len(SEEDS)}")
    for s, run in zip(SEEDS, runs):
        if run.get("ds") != ds:
            raise ValueError(f"cpgate run {s}: dataset {run.get('ds')!r}, expected {ds!r} (a run of another "
                             "dataset)")
        if run.get("seed") != s:
            raise ValueError(f"cpgate run {s} of {ds}: seed {run.get('seed')!r}, expected {s}")
        if run.get("gate_norm") != CPPR_GATE:
            raise ValueError(f"cpgate run {s} of {ds}: gate {run.get('gate_norm')!r}, expected the "
                             f"{CPPR_GATE!r} gate")


def load_cppr_runs(ds: str, inputs: Inputs) -> list:
    runs = [inputs.json(CPPR_FILE.format(ds=ds, seed=s)) for s in SEEDS]
    check_cppr_runs(ds, runs)
    return runs


def cppr_seed_block(runs: list, boot: Boot) -> dict:
    """Seed-level statistics of the whitewash gains and the test-MCC change (paired design: same seeds)."""
    seeds = [int(r["seed"]) for r in runs]
    vpe = [float(r["inject_then_erase_whitewash"]["gain_plain_vpe"]) for r in runs]
    cppr = [float(r["inject_then_erase_whitewash"]["gain_cp_vpe"]) for r in runs]
    dmcc = [float(r["direct_laundering"]["d_t1_mcc_cp_vs_plain"]) for r in runs]
    b = Block("cppr_seed", boot)
    for field, values in (("whitewash_plain_vpe", vpe), ("whitewash_cppr", cppr),
                          ("whitewash_paired_diff", [a - c for a, c in zip(vpe, cppr)]),
                          ("d_mcc_cppr_minus_plain", dmcc)):
        b.stat(field, {s: [x] for s, x in zip(seeds, values)}, support=None)    # signed: not clipped (R31)
    for field, values in (("whitewash_plain_vpe", vpe), ("whitewash_cppr", cppr)):
        b.exact(f"{field}_sd", float(np.std(values, ddof=1)), "sample sd over seeds (no interval)")
    b.exact("d_mcc_cppr_minus_plain_min", min(dmcc), "minimum over seeds (no interval)")
    b.exact("d_mcc_cppr_minus_plain_max", max(dmcc), "maximum over seeds (no interval)")
    b.extra.update(label=PRE_REVISION, seeds=seeds, paired_t=paired_t(vpe, cppr))
    return b.to_dict()


# ------------------------------------------------------------------------ analysis, method, provenance, CLI
def analyse(datasets, inputs: Inputs, n_boot: int = N_BOOT) -> dict:
    coverage, cppr = {}, {}
    for ds in datasets:
        print(f"[{ds}] tier coverage and CPPR seed statistics", flush=True)
        D = C.load(ds)
        for rel in [f"configs/{ds}.json"] + [D.cfg[k] for k in ("data_path", "edge_path") if D.cfg.get(k)]:
            inputs.path(rel)
        boot = Boot(ds, n_boot)
        coverage[ds] = coverage_block(tier_coverage(C.train_split(C.full_graph(D), D.tt), D.n), boot)
        cppr[ds] = cppr_seed_block(load_cppr_runs(ds, inputs), boot)
    return dict(tier_coverage=coverage, single_warning=single_warning_block(), cppr_seed_stats=cppr)


METHOD = dict(
    tier_coverage="Training graph of each CPPR network (keeper.laundering.core load, full_graph, "
                  "train_split with the dataset's tt). Tiers from keeper.credibility.tiers.legacy_bfs (the "
                  "legacy seeded tiers, moved verbatim from cp_vpe_gpu.compute_credibility_levels). Seeds "
                  "S0: accounts with more positive than negative in-edges in snapshots 0 to 4. D: S0 plus "
                  "every account that one pass over the edges in stored order reaches by a positive rating "
                  "from a member of D (review_tiers.py), checked against legacy_bfs. Active: accounts with "
                  "a training edge at either end. Late joiners: active accounts first active after "
                  "snapshot 4; late_in_D_share is the share of them in D. Tier shares are over the active "
                  "accounts. Every value is a census of the training graph, so ci95 is null.",
    single_warning="revoked_no_expiry: Eq. 8 (keeper.scem.scem.revocation_curve_C), a constant hazard from "
                   "age lag + L_f on, with no expiry and no closure (the paper's curve and census). "
                   "revoked_engine_rules: the engine's rules, with the hazard live for H snapshots from "
                   "maturity and the record closed at A (erlang_Q_P_np and segment_release_increment_np); "
                   "durable = 1 - revoked. hazard_map: one warning's hazard by the engine's formula (h = L_h "
                   "xi, nu = softplus(h) - log 2, lam^sigma = lambda_max (1 - exp(-gamma_sigma nu / "
                   "lambda_max)), gamma = (1.0, 0.8), each witness capped at c = 1). Its revoked fractions "
                   "are given per channel and mixed by a rating's routing: psi = (0.8, 0.2) for a positive "
                   "and (0.2, 0.8) for a negative rating, as the engine mixes them. The reliabilities are "
                   "1.0 (a fully reliable witness) and the engine's floors, 0.35 up to snapshot 4 and 0.20 "
                   "after it. witnesses_to_reach: the fewest fully reliable witnesses whose channel-1 hazard "
                   "reaches 1.9. The constants are the defaults of run_scem_streaming_weights_np. "
                   "Deterministic, so there is no interval.",
    cppr_seed_stats="Pre-revision runs (superseded by E3): results/laundering/cpgate_count_<ds>_s<seed>.json, "
                    "seeds 0 to 4, count gate. Whitewash gains: inject_then_erase_whitewash.gain_plain_vpe "
                    "(Plain VPE) and gain_cp_vpe (CPPR), each a mean over one seed's targets. The paired "
                    "difference is Plain VPE minus CPPR, seed by seed; paired_t gives its mean, sd, t = mean "
                    "/ (sd / sqrt(n)), df = n - 1 = 4 and the two-sided p. d_mcc_cppr_minus_plain: "
                    "direct_laundering.d_t1_mcc_cp_vs_plain, the test T1 MCC of CPPR minus that of Plain "
                    "Erasure.",
    ci95="CPPR seed fields follow ruling R30. ci95 is the hull of two intervals. ci95_bootstrap is the "
         "percentile interval of provenance.n_boot bootstrap resamples of the seeds (10,000 by default); the "
         "runs keep one value per seed, so the crossed design of ruling R29 reduces to resampling seeds, and "
         "ci95_nested equals it in law. ci95_seed_t is the t-interval over the per-seed values with df = "
         "n_seeds - 1. The fields are signed, so ruling R31 clips none of them. sd, minimum, maximum, census "
         "and closed-form values have no interval.")


def provenance(inputs: Inputs, n_boot: int) -> dict:
    import scipy
    return dict(
        **git_state(),                                          # commit, dirty flag, uncommitted paths
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        host=socket.gethostname(), argv=list(sys.argv), python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), n_boot=int(n_boot),
        boot_seed=BOOT_SEED, code_sha256=C.CODE_SHA256_AT_IMPORT,
        module_sha256={rel: sha256_file(os.path.join(C.ROOT, rel)) for rel in MODULE_FILES},
        inputs_sha256=dict(sorted(inputs.sha256.items())))


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Defense re-analysis of the stored results (CPU only).")
    ap.add_argument("--datasets", nargs="+", default=list(NETWORKS), choices=NETWORKS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "defense_extra.json"))
    ap.add_argument("--md", default=os.path.join(ANALYSIS, "defense_extra.md"))
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    from keeper.analysis.defense_extra_md import render_markdown
    a = parse(argv)
    out_path = C.result_path(os.path.abspath(a.out), a.force)      # refuse before the work, not after it
    md_path = C.result_path(os.path.abspath(a.md), a.force)
    inputs = Inputs()
    result = analyse(a.datasets, inputs, a.n_boot)
    result["method"] = METHOD
    result["provenance"] = provenance(inputs, a.n_boot)
    for path in (out_path, md_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    C.dump(result, out_path, force=a.force)
    text = render_markdown(json.loads(json.dumps(C._finite(result), default=C._json_default)))
    with open(md_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path} and {md_path}")
    return result


if __name__ == "__main__":
    main()
