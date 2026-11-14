"""Synthetic E3 and E4 results and target dumps for tests/test_cppr_confirm.py (helpers, not collected).

The files follow the CPPR driver's output schema, read from keeper/cppr/cp_vpe_gpu.py (run_single_seed),
keeper/cppr/protocol.py (config_block, sybil_admission) and keeper/cppr/dumps.py (TARGET_COLUMNS, dump_table); the GPU
smoke output .superpowers/sdd/2026-10-26-keeper-tdsc-revision/gate-4.1b/e3_smoke.json has the same keys. They are
written where keeper.cluster.cppr_tasks puts the real runs, under a temporary results root:
  results/cppr/{e3,e4}/<stem>.json and results/cppr/{e3,e4}/targets/<stem>_targets.json.

Every value is a deterministic function of the run and the target (Model), so that a test knows each endpoint's sign:
Plain Erasure launders (+0.05), Plain VPE keeps most of the fakes' effect (+0.03 whitewash, -0.03 bad-mouth), CPPR
removes nearly all of it (+-0.001), the control arm only adds noise, and in E4 BFS gains more than capped BTP in every
cell. The hand-computed endpoint tests use small explicit tables instead.
"""
import copy
import hashlib
import json
import math
import os
import zlib

import numpy as np

from keeper.analysis import cppr_runs as R
from keeper.cluster import cppr_tasks as T
from keeper.cluster import protocol_settings as PS
from keeper.cppr import dumps as DP
from keeper.cppr.protocol import ARMS, ATTACK_RNG_ENTROPY, CAP_TOL, ENDORSER_MIN_TIER

N_EDGE, N_ACC = 15, 5
EDGE_POOL = range(0, 40)            # edge-level targets of a seed: 15 of these (they recur across seeds)
ACCOUNT_POOL = range(40, 60)        # account-level targets: 5 of these
VICTIM_POOL = range(100, 140)       # bad-mouth victims: 15 of these
NOISE = 0.004                       # every gain carries uniform noise in [-NOISE/2, NOISE/2]
PREREG_TEXT = "# KEEPER revision: protocol of the confirmatory experiments (synthetic copy for tests)\n"

BASE_GAIN = {("direct", "plain", "edge"): 0.05, ("direct", "vpe", "edge"): 0.01, ("direct", "cppr", "edge"): 0.001,
             ("direct", "plain", "account"): 0.04, ("direct", "vpe", "account"): 0.01,
             ("direct", "cppr", "account"): 0.001, ("whitewash", "vpe", "edge"): 0.03,
             ("whitewash", "cppr", "edge"): 0.001, ("badmouth", "vpe", "trusted"): -0.03,
             ("badmouth", "cppr", "trusted"): -0.001}           # control rows: noise only
E4_GAIN = {T.BFS: 0.02, T.CAPPED: 0.002, T.UNCAPPED: 0.015}     # plus 0.002 per index in E4_CELLS
MCC = {"plain": 0.28, "vpe": 0.29, "cppr": 0.30}
UNGATED_MCC = 0.27


def unit(*parts) -> float:
    """A deterministic number in [0, 1) from the parts."""
    return zlib.crc32(repr(parts).encode()) / 2 ** 32


def targets(ds: str, seed: int) -> dict:
    """The seed's edge-level targets, account-level targets and bad-mouth victims, sorted."""
    rng = np.random.default_rng((PS.NETWORKS.index(ds), seed))
    pick = lambda pool, k: sorted(int(v) for v in rng.choice(np.asarray(pool), k, replace=False))  # noqa: E731
    return dict(edge=pick(EDGE_POOL, N_EDGE), account=pick(ACCOUNT_POOL, N_ACC), trusted=pick(VICTIM_POOL, N_EDGE))


class Model:
    """Synthetic base reputations (one base model per tier setting, dataset and seed) and gains per row."""

    def r0(self, tier: str, ds: str, seed: int, target: int) -> float:
        return 0.2 + 0.6 * unit("r0", tier, ds, seed, target)

    def gain(self, key, arm: str, method: str, kind: str, target: int) -> float:
        noise = NOISE * (unit("gain", key.stem, arm, method, kind, target) - 0.5)
        if key.family == "e4":
            return E4_GAIN[key.tier] + 0.002 * T.E4_CELLS.index(key.cell) + noise
        return BASE_GAIN.get((arm, method, kind), 0.0) + noise


def calibration_fields(ds: str, eps: float, units) -> dict:
    """The eps, K, theta0, edge_cap_units, edge_cap, n_ref, n_seeds and t_seed of a frozen BTP calibration."""
    n_ref = PS.N_REF_EXPECTED[ds]
    return dict(eps=eps, K=PS.K_EXPECTED[ds], theta0=eps / n_ref, edge_cap_units=units,
                edge_cap=None if units is None else units * eps / n_ref, n_ref=n_ref, n_seeds=1188, t_seed=4)


def tiers_config(key) -> dict:
    """config.tiers as the driver records it after Tasks 3.1b and 4.1c: the run's tier calibration and its
    eligibility guard neg_from ("trusted" under BTP, null under BFS, which has no calibration)."""
    tiers = dict(method="bfs", eps=None, K=None, theta0=None, edge_cap_units=None, edge_cap=None, neg_from=None,
                 n_ref=None, n_seeds=None, t_seed=None, calibration_used_for=None)
    if key.tier == T.BFS:
        return tiers
    eps = (PS.EPS_CAPPED if key.tier == T.CAPPED else PS.EPS_UNCAPPED)[key.ds]
    tiers.update(method="btp", neg_from="trusted", calibration_used_for="tiers",
                 **calibration_fields(key.ds, eps, PS.EDGE_CAP_UNITS if key.tier == T.CAPPED else None))
    return tiers


def config(key, prereg_sha: str) -> dict:
    cell, e3 = key.cell, key.family == "e3"
    reference = None if e3 else calibration_fields(key.ds, PS.EPS_CAPPED[key.ds], PS.EDGE_CAP_UNITS)
    attack = dict(attack_edges=0 if e3 else cell.g, endorser_select="random" if e3 else cell.endorsers,
                  sybil_topology="ring" if e3 else cell.topology, k_fake=3 if e3 else cell.s,
                  endorser_min_tier=ENDORSER_MIN_TIER, endorser_candidates="honest accounts other than the arm's targets",
                  endorser_assignment="ranked" if not e3 and cell.endorsers == "max_trust" else "common_pool",
                  reference_calibration=reference,
                  random_stream=f"numpy default_rng(({ATTACK_RNG_ENTROPY}, seed, index of the arm in {list(ARMS)}))")
    methods = (["plain_erasure", "plain_vpe", "cppr"] if e3
               else ["plain_vpe", "cppr"] if (cell, key.tier) == T.E4_PAIRED_RUN else ["cppr"])
    return dict(tiers=tiers_config(key), attack=attack,
                arms=["direct", "whitewash", "badmouth", "control"] if e3 else ["whitewash"], methods=methods,
                hold_test_snapshot=True, ungated_baseline=e3 and key.tier == T.CAPPED, gate_norm="count",
                preregistration=dict(path="docs/preregistration.md", sha256=prereg_sha))


def target_rows(key, model: Model) -> list:
    """Rows of keeper.cppr.dumps.TARGET_COLUMNS, arm by arm as the driver writes them."""
    tg, rows = targets(key.ds, key.seed), []

    def add(arm, method, kind):
        for t in tg[kind]:
            r0 = model.r0(key.tier, key.ds, key.seed, t)
            r_after = r0 + model.gain(key, arm, method, kind, t)
            rows.append([arm, method, kind, t, r0, r_after, r_after - r0, 2, 1, 0, 1, 1, False])

    if key.family == "e4":
        for method in (("vpe", "cppr") if (key.cell, key.tier) == T.E4_PAIRED_RUN else ("cppr",)):
            add("whitewash", method, "edge")
        return rows
    for method in ("plain", "vpe", "cppr"):
        add("direct", method, "edge")
        add("direct", method, "account")
    for arm, kind in (("whitewash", "edge"), ("badmouth", "trusted"), ("control", "edge"), ("control", "trusted")):
        for method in ("vpe", "cppr"):
            add(arm, method, kind)
    return rows


def admission(key) -> dict:
    """sybil_admission: every capped group admits min(s, guarantee) fakes, BFS and uncapped groups all s (g > 0)."""
    g, s = (0, 3) if key.cell is None else (key.cell.g, key.cell.s)
    guarantee = math.floor(PS.K_EXPECTED[key.ds] * g * PS.EDGE_CAP_UNITS + CAP_TOL) if key.tier == T.CAPPED else None
    per_group = 0 if g == 0 else s if guarantee is None else min(s, guarantee)
    arms = ("whitewash", "badmouth") if key.family == "e3" else ("whitewash",)
    return {arm: dict(n_fakes=N_EDGE * s, n_admitted=N_EDGE * per_group, admitted_per_group=[per_group] * N_EDGE,
                      fake_tier_counts=[N_EDGE * (s - per_group), N_EDGE * per_group, 0, 0],
                      endorsers=None if g == 0 else [[1000 + i * g + j for j in range(g)] for i in range(N_EDGE)],
                      snap_f=[5 + i % 15 for i in range(N_EDGE)], guarantee_per_group=guarantee,
                      endorsers_in_D_at_attack=None, pool_size=None, pool_size_untimed=None,
                      share_removed_by_timing=None, timing_applied=None)
            for arm in arms}


def metrics(mcc: float) -> dict:
    """keeper.laundering.core.test_metrics: top-level mcc, auc and n are Task 1's."""
    t1 = dict(mcc=mcc, auc=0.78, n=642)
    return dict(t1, t1=t1, t2=dict(mcc=-0.1, auc=0.5, n=22))


def held_metrics(key) -> dict:
    jitter = lambda *p: 0.01 * (unit("mcc", key.stem, *p) - 0.5)  # noqa: E731
    out = dict(base=metrics(0.3 + jitter("base")))
    if key.family == "e3":
        out["direct"] = {m: metrics(MCC[m] + jitter("direct", m)) for m in ("plain", "vpe", "cppr")}
        for arm in ("whitewash", "badmouth", "control"):
            out[arm] = {m: metrics(MCC[m] + jitter(arm, m)) for m in ("vpe", "cppr")}
    else:
        out["whitewash"] = {m: metrics(MCC[m] + jitter("whitewash", m))
                            for m in (("vpe", "cppr") if (key.cell, key.tier) == T.E4_PAIRED_RUN else ("cppr",))}
    return out


def provenance() -> dict:
    return dict(host="test-host", utc="2026-10-28T00:00:00Z", argv=["keeper/cppr/cp_vpe_gpu.py"], device="cuda:0",
                gpu_name="NVIDIA RTX A6000", deterministic=True, cublas_workspace=":4096:8",
                code_sha256="c" * 64, git_commit="a" * 40, git_dirty=False, git_untracked=0, git_status=[])


def build_run(root, key, model: Model, prereg_sha: str) -> tuple:
    """(result, dump) of one run, as the driver writes them (the dump through keeper.cppr.dumps.dump_table)."""
    rows = target_rows(key, model)
    cfg, prov = config(key, prereg_sha), provenance()
    result = dict(ds=key.ds, seed=key.seed, epochs=150, rho=0.02, granularity=1, k_fake=cfg["attack"]["k_fake"],
                  gate_norm="count", temporal_encoder="SCEM", base_t1=metrics(0.3)["t1"], config=cfg,
                  sybil_admission=admission(key), test_metrics=held_metrics(key),
                  dumps=dict(targets=dict(path=key.dump_file(str(root)), n_rows=len(rows))), runtime_s=1.0,
                  provenance=prov)
    if key.family == "e3" and key.tier == T.CAPPED:
        result["ungated_baseline"] = metrics(UNGATED_MCC + 0.01 * (unit("ungated", key.stem) - 0.5))
    if key.family == "e4":                       # Task 4.1c: every E4 run uses the base-model cache
        result["base_model"] = dict(from_cache=key.cell != T.E4_CELLS[0], key=f"{key.tier}{key.seed}".ljust(64, "0"),
                                    gpu_name=prov["gpu_name"], torch="2.4.0", created_utc=prov["utc"],
                                    code_commit=prov["git_commit"])
    dump = DP.dump_table(DP.TARGET_COLUMNS, rows, dict(result=f"{key.stem}.json", config=copy.deepcopy(cfg),
                                                       provenance=copy.deepcopy(prov)))
    return result, dump


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def write_prereg(root) -> tuple:
    """(path, sha256) of the synthetic protocol file root/docs/preregistration.md."""
    path = os.path.join(str(root), "docs", "preregistration.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(PREREG_TEXT)
    return path, hashlib.sha256(PREREG_TEXT.encode()).hexdigest()


def build_results(root, networks=("bitcoinalpha",), model: Model | None = None, mutate=None, skip=()) -> str:
    """Write every expected run of ``networks`` under root (except the stems in ``skip``) and the protocol file;
    ``mutate(key, result, dump)`` may change a run before it is written. Returns the protocol file's path."""
    model = model or Model()
    prereg, sha = write_prereg(root)
    for key in R.expected_runs(networks):
        if key.stem in skip:
            continue
        result, dump = build_run(root, key, model, sha)
        if mutate is not None:
            mutate(key, result, dump)
        write_json(key.result_file(str(root)), result)
        write_json(key.dump_file(str(root)), dump)
    return prereg


def rows_of(dump: dict, arm: str, method: str, kind: str) -> list:
    """The dump rows (lists, mutable in place) of one arm, method and kind."""
    return [row for row in dump["rows"] if row[:3] == [arm, method, kind]]
