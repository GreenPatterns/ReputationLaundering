"""The E3/E4 run set of the confirmatory CPPR analysis (Task 4.2b): which runs are expected, how they are loaded, the
integrity checks each must pass, and the replay of the driver's target draws.

The expected runs are keeper.cluster.cppr_tasks' task lists: <results root>/results/cppr/{e3,e4}/<stem>.json and their
target dumps .../targets/<stem>_targets.json (keeper.cppr.dumps). The registered values come from
keeper.cluster.protocol_settings and from the protocol (docs/preregistration.md, Sections 2.1, 2.2, 2.5, 2.6 and 6).

Integrity (registered_settings, integrity_failures). A run is used only when every recorded setting equals the
registered one: ds and seed (the file name's); epochs, rho, the snapshot granularity, k_fake and gate_norm; the arms and
methods; the attack (attack edges, endorser selection, topology); config.tiers' method, eps, K, n_ref, edge_cap_units
and neg_from (the credible-negatives guard of ruling R56; null under BFS); an E4 run's reference capped calibration of
the endorser protocol (--pool-eps); hold_test_snapshot; the protocol file's sha256; a recorded git commit with no
modified tracked file and no untracked file. Its target dump must belong to it (result name, config, row count) and hold
one finite row per arm, method, kind and target with one r0 per target. Every violated field is listed; a run that fails
is excluded and counted, never used.

Replay (seed_targets). The driver (keeper.cppr.cp_vpe_gpu.run_single_seed) draws a seed's edge-level targets with its
workload on the training graph, then, from default_rng(3000 + seed), one snapshot per whitewash group
(keeper.cppr.arm_graphs.build_inject_then_erase_graph) and the bad-mouth victims from the trusted cohort. Plain VPE and
CPPR draw nothing from that stream at eps = inf, which the driver passes. seed_targets replays these draws without
training; the analysis compares the replay with every E3 run's dump and drops the stored seeds' targets in the
unused-target sensitivity (Section 2.1, ruling R62).
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from keeper.cluster import cppr_tasks as T
from keeper.cluster import protocol_settings as PS
from keeper.cppr import dumps as DP
from keeper.cppr.protocol import FIRST_FAKE_SNAPSHOT
from keeper.cppr.protocol import METHODS as E3_METHODS     # E3 passes no --methods: the driver's default, all three
from keeper.laundering import core as C

STORED_SEEDS = range(5)          # seeds 0-4 produced the stored results (Section 2.1)
# The driver's code defaults, which E3 and E4 take (Section 2.5): targets, rho, granularity, region size, gate.
N_EDGE, N_ACC, RHO, GRANULARITY, E3_K_FAKE, GATE_NORM = 15, 5, 0.02, 1, 3, "count"
TARGET_SEED, MAX_NEG, TRUSTED_MIN_POS = 3000, 3, 3       # run_single_seed: default_rng(3000 + seed), the cohorts
E3_ARMS = tuple(PS.E3_ARMS.split(","))
E4_ARMS = tuple(PS.E4_ARMS.split(","))
TIERS = (PS.CAPPED, PS.BFS, PS.UNCAPPED)


def cell_label(cell: PS.Cell) -> str:
    """An E4 cell as the task ids spell it, e.g. rand_ring_k3_g1."""
    return f"{T.ENDORSER_TAGS[cell.endorsers]}_{cell.topology}_k{cell.s}_g{cell.g}"


@dataclass(frozen=True)
class RunKey:
    """One expected run: family (e3 or e4), tier setting (btpc, bfs or btpu), dataset, seed and, for E4, its cell."""
    family: str
    tier: str
    ds: str
    seed: int
    cell: PS.Cell | None = None

    @property
    def stem(self) -> str:
        if self.family == "e3":
            return f"e3_{self.tier}_{self.ds}_s{self.seed}"
        return f"e4_{self.tier}_{cell_label(self.cell)}_{self.ds}_s{self.seed}"

    def result_file(self, root: str) -> str:
        return os.path.join(root, "results", "cppr", self.family, f"{self.stem}.json")

    def dump_file(self, root: str) -> str:
        return os.path.join(root, "results", "cppr", self.family, "targets", f"{self.stem}_targets.json")


@dataclass
class Run:
    """A loaded run: its key, result JSON and indexed target dump, (arm, method, kind) -> {target: row}."""
    key: RunKey
    result: dict
    index: dict


def expected_runs(networks=PS.NETWORKS) -> list:
    """The registered runs of ``networks``: E3 capped BTP and BFS on SEEDS_E3 and uncapped BTP on SEEDS_5; E4 every
    cell under BFS and capped BTP, and E4_UNCAPPED_CELLS also under uncapped BTP, on SEEDS_5."""
    keys = []
    for ds in networks:
        for seed in PS.SEEDS_E3:
            keys += [RunKey("e3", tier, ds, seed)
                     for tier in (PS.CAPPED, PS.BFS) + ((PS.UNCAPPED,) if seed in PS.SEEDS_5 else ())]
        for seed in PS.SEEDS_5:
            keys += [RunKey("e4", tier, ds, seed, cell) for cell in PS.E4_CELLS
                     for tier in (PS.BFS, PS.CAPPED) + ((PS.UNCAPPED,) if cell in PS.E4_UNCAPPED_CELLS else ())]
    return keys


def check_run_set(keys: list, results_root: str, networks) -> None:
    """Raise unless ``keys`` are exactly the task lists' runs of ``networks`` (ids and output paths)."""
    def dataset_of(task_id: str) -> str:              # ids end in _<ds>_s<seed>
        return next((ds for ds in PS.NETWORKS if task_id.rsplit("_s", 1)[0].endswith(f"_{ds}")), "")
    rows = [r for r in T.e3_tasks(results_root) + T.e4_tasks(results_root) if dataset_of(r.id) in networks]
    if {k.stem: k.result_file(results_root) for k in keys} != {r.id: r.done_path for r in rows}:
        raise RuntimeError("the expected run set differs from keeper.cluster.cppr_tasks' task lists")


class Manifest:
    """Every file the analysis reads, with the sha256 of the bytes it parsed; keyed by the path relative to the results
    root, else to the repository root, else absolute."""

    def __init__(self, results_root: str):
        self.roots = (os.path.abspath(results_root), C.ROOT)
        self.sha256 = {}

    def key(self, path: str) -> str:
        path = os.path.abspath(path)
        for root in self.roots:
            rel = os.path.relpath(path, root)
            if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
                return rel
        return path

    def read(self, path: str) -> bytes:
        with open(path, "rb") as f:
            data = f.read()
        self.sha256[self.key(path)] = hashlib.sha256(data).hexdigest()
        return data

    def add(self, path: str) -> str:
        self.read(path)
        return self.sha256[self.key(path)]

    def json(self, path: str):
        return json.loads(self.read(path))


# ------------------------------------------------------------------------ integrity checks
def calibration(eps: float, units, ds: str) -> dict:
    """The registered eps, K, n_ref and edge_cap_units of a BTP calibration on ``ds`` (Section 2.2)."""
    return dict(eps=eps, K=PS.K_EXPECTED[ds], n_ref=PS.N_REF_EXPECTED[ds], edge_cap_units=units)


def expected_tiers(tier: str, ds: str) -> dict:
    """The registered config.tiers of a tier setting: BFS takes no calibration and no guard (Section 2.2, R56)."""
    if tier == PS.BFS:
        return dict(method="bfs", eps=None, K=None, n_ref=None, edge_cap_units=None, neg_from=None)
    capped = tier == PS.CAPPED
    return dict(method="btp", neg_from=PS.NEG_FROM,
                **calibration((PS.EPS_CAPPED if capped else PS.EPS_UNCAPPED)[ds],
                              PS.EDGE_CAP_UNITS if capped else None, ds))


def registered_settings(key: RunKey, prereg_sha256: str | None) -> dict:
    """Field path (dots between keys) -> the value the protocol registers for run ``key``."""
    e3, cell = key.family == "e3", key.cell
    k_fake = E3_K_FAKE if e3 else cell.s
    methods = (E3_METHODS if e3 else
               PS.E4_PAIRED_METHODS.split(",") if (cell, key.tier) == PS.E4_PAIRED_RUN else PS.E4_METHODS.split(","))
    want = {"ds": key.ds, "seed": key.seed, "epochs": PS.EPOCHS, "rho": RHO, "granularity": GRANULARITY,
            "k_fake": k_fake, "gate_norm": GATE_NORM, "config.gate_norm": GATE_NORM,
            "config.arms": list(E3_ARMS if e3 else E4_ARMS), "config.methods": list(methods),
            "config.hold_test_snapshot": True, "config.attack.k_fake": k_fake,
            "config.attack.attack_edges": 0 if e3 else cell.g,
            "config.attack.endorser_select": "random" if e3 else cell.endorsers,
            "config.attack.sybil_topology": "ring" if e3 else cell.topology,
            "provenance.git_dirty": False, "provenance.git_untracked": 0}
    want.update({f"config.tiers.{f}": v for f, v in expected_tiers(key.tier, key.ds).items()})
    if e3:
        want["config.attack.reference_calibration"] = None
    else:                                       # --pool-eps: the reference capped calibration (Sections 2.2 and 6)
        want.update({f"config.attack.reference_calibration.{f}": v
                     for f, v in calibration(PS.EPS_CAPPED[key.ds], PS.EDGE_CAP_UNITS, key.ds).items()})
    if prereg_sha256 is not None:
        want["config.preregistration.sha256"] = prereg_sha256
    return want


def lookup(obj, path: str):
    """obj[a][b]... for the path "a.b..."; None where a key is missing."""
    for part in path.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


def _same(got, want) -> bool:
    return got == want and isinstance(got, bool) == isinstance(want, bool)     # 0 is not False


def _dump_failures(key: RunKey, result: dict, dump: dict, kind: str = "targets") -> list:
    """The ways a dump of ``kind`` (targets or verdicts) fails to belong to its run; only key.stem is read."""
    recorded = (result.get("dumps") or {}).get(kind) or {}
    rows, n_rows = dump.get("rows"), dump.get("n_rows")
    checks = (("dump.result", dump.get("result") == f"{key.stem}.json",
               f"is {dump.get('result')!r}, not {key.stem}.json"),
              ("dump.config", dump.get("config") == result.get("config"), "differs from the result's config"),
              ("dump.n_rows", isinstance(rows, list) and n_rows == len(rows) == recorded.get("n_rows"),
               f"{n_rows!r} differs from its rows or from dumps.{kind}.n_rows {recorded.get('n_rows')!r}"),
              (f"dumps.{kind}.path", os.path.basename(str(recorded.get("path"))) == f"{key.stem}_{kind}.json",
               f"{recorded.get('path')!r} does not name {key.stem}_{kind}.json"))
    return [dict(field=field, detail=f"{field} {text}") for field, ok, text in checks if not ok]


def integrity_failures(key: RunKey, result: dict, dump: dict, prereg_sha256: str | None) -> list:
    """The integrity failures of one run, dict(field, detail) each; [] when it is the registered run it claims to
    be."""
    out = [dict(field=path, detail=f"{path} is {lookup(result, path)!r}, not the registered {want!r}")
           for path, want in registered_settings(key, prereg_sha256).items() if not _same(lookup(result, path), want)]
    commit = lookup(result, "provenance.git_commit")
    if not (isinstance(commit, str) and commit):
        out.append(dict(field="provenance.git_commit", detail=f"provenance.git_commit is {commit!r}, not a commit"))
    return out + _dump_failures(key, result, dump)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def index_rows(dump: dict) -> tuple:
    """((arm, method, kind) -> {target: row}, failure messages) of a target dump with keeper.cppr.dumps.TARGET_COLUMNS.
    A row of the wrong length or with a non-finite r0 or gain is left out; duplicates and differing r0 are failures."""
    columns, index, r0, failures = list(DP.TARGET_COLUMNS), {}, {}, []
    if dump.get("columns") != columns:
        return {}, [f"the target dump's columns {dump.get('columns')} are not {columns}"]
    for row in dump.get("rows") or []:
        if not isinstance(row, list) or len(row) != len(columns):
            failures.append(f"a target dump row has {len(row) if isinstance(row, list) else '?'} fields")
            continue
        rec = dict(zip(columns, row))
        where = f"{rec['arm']}/{rec['method']}/{rec['kind']} target {rec['target']}"
        if not (_finite(rec["r0"]) and _finite(rec["gain"])):
            failures.append(f"{where}: r0 {rec['r0']!r} or gain {rec['gain']!r} is not finite")
            continue
        table = index.setdefault((rec["arm"], rec["method"], rec["kind"]), {})
        target = int(rec["target"])
        if target in table:
            failures.append(f"{where}: duplicate row")
            continue
        table[target] = rec
        if r0.setdefault(target, rec["r0"]) != rec["r0"]:
            failures.append(f"{where}: r0 {rec['r0']!r} differs from the target's r0 {r0[target]!r} in another row")
    return index, failures


def load_run(key: RunKey, results_root: str, manifest: Manifest, prereg_sha256: str | None) -> tuple:
    """(run or None, missing reason or None, failures as dict(field, detail)) of one expected run. A run with a
    failure must not be used."""
    result_file, dump_file = key.result_file(results_root), key.dump_file(results_root)
    if not os.path.exists(result_file):
        return None, "result missing", []
    if not os.path.exists(dump_file):
        return None, "target dump missing", []
    try:
        result, dump = manifest.json(result_file), manifest.json(dump_file)
    except (OSError, ValueError) as err:
        return None, None, [dict(field="file", detail=f"cannot read the result or its dump: {err}")]
    if not (isinstance(result, dict) and isinstance(dump, dict)):
        return None, None, [dict(field="file", detail="the result or its dump is not a JSON object")]
    index, row_failures = index_rows(dump)
    failures = integrity_failures(key, result, dump, prereg_sha256)
    return Run(key, result, index), None, failures + [dict(field="dump.rows", detail=d) for d in row_failures]


# ------------------------------------------------------------------------ the driver's target draws, replayed
def workload_function():
    """The workload function the CPPR driver calls (keeper.cppr.cp_vpe_gpu.run_single_seed)."""
    from keeper.cppr import vpe as V
    from keeper.cppr import vpe_frontier
    return V.workload if hasattr(V, "workload") else vpe_frontier.workload


def seed_targets(ds: str, seeds, manifest: Manifest | None = None) -> dict:
    """seed -> dict(edge=[edge-level targets], trusted=[bad-mouth victims]) that the CPPR driver draws for ``ds`` and
    that seed, replayed without training in run_single_seed's order: the workload on the training graph (target seed
    3000 + seed, honest background default_rng(int(1e6 rho) + 17 seed)); then, from default_rng(3000 + seed), one
    snapshot per whitewash group and the victims from the trusted cohort (at least three positive and no negative
    training in-edges). The dataset files read are added to ``manifest``."""
    D = C.load(ds)
    if manifest is not None:
        for rel in [f"configs/{ds}.json"] + [D.cfg[k] for k in ("data_path", "edge_path") if D.cfg.get(k)]:
            manifest.add(os.path.join(C.ROOT, rel))
    Tr = C.train_split(C.full_graph(D), D.tt)
    cohort, neg, pos = C.pick_targets(Tr, max_neg=MAX_NEG, strict=True)
    trusted = np.array(sorted(v for v, rows in pos.items() if len(rows) >= TRUSTED_MIN_POS and not neg.get(v)),
                       dtype=np.int64)
    workload, out = workload_function(), {}
    for seed in seeds:
        honest_rng = np.random.default_rng(int(1e6 * RHO) + 17 * seed)
        _, edge_t, _ = workload(Tr, cohort, neg, N_EDGE, N_ACC, RHO, TARGET_SEED + seed, honest_rng)
        rng = np.random.default_rng(TARGET_SEED + seed)
        for _ in edge_t:                                # build_inject_then_erase_graph: snap_f per whitewash group
            rng.integers(FIRST_FAKE_SNAPSHOT, D.tt)
        victims = rng.choice(trusted, size=min(len(edge_t), len(trusted)), replace=False)
        out[int(seed)] = dict(edge=[int(v) for v in edge_t], trusted=[int(v) for v in victims])
    return out


def stored_seed_targets(ds: str, manifest: Manifest | None = None) -> dict:
    """dict(edge=set, trusted=set): the edge-level targets and bad-mouth victims of the stored seeds 0-4 on ``ds``."""
    drawn = seed_targets(ds, STORED_SEEDS, manifest).values()
    return {kind: set().union(*(set(d[kind]) for d in drawn)) for kind in ("edge", "trusted")}
