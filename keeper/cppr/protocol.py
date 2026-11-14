"""The run protocol of the CPPR driver keeper.cppr.cp_vpe_gpu (Task 3.2 of the 2026-10 revision).

RunProtocol holds the options that the stored runs did not have; its defaults reproduce them (is_legacy).
* Tier method (--tiers). CredibilityTiers is the run's one tier function, used for cred_levels on the base
  graph and for cred_inj on every injected graph. bfs is the legacy seeded endorsement of Eq. (14). btp is BTP
  (upgrade M-A) under one calibration frozen on the run's honest base training graph (ruling R32), with KEEPER's
  per-endorsement cap edge_cap_units = 2 by default (rulings R37, R38) and the credible-negatives guard by
  default (--btp-neg-from trusted, ruling R56): only trusted raters' negative ratings count against
  eligibility, so fakes without attack edges cannot demote anyone; all is the ablation.
* Attack edges (--attack-edges g, --endorser-select, --sybil-topology, --pool-eps). Each target group's fakes
  receive g positive ratings from distinct honest accounts other than the arm's targets. An EndorserPicker chooses
  them once the group's snap_f is drawn, from the EndorserInputs that endorser_inputs reads off the honest base
  training graph. Task 4.1c: the endorser protocol uses a reference capped BTP calibration (eps = --pool-eps,
  KEEPER's cap), frozen on that graph, whatever the run's tiers.
  - random (ruling R43): one pool for every tier method (common_pool): BFS tier >= 2, reference tier >= 2, not a
    target, already in legacy's trusted set D when the pass reads the group's attack edges (ruling R39), and
    reference inflow B(u) >= theta0. Runs of any tier method with the same seed, arm and attack flags therefore
    draw the same endorsers.
  - max_trust (ruling R44): group i (from 0) takes the accounts ranked i g + 1 .. (i + 1) g by B(u) (ruling
    R33b) among the accounts other than the arm's targets: the reference B(u) under BFS and capped tiers, the
    run's own uncapped B(u) under uncapped tiers.
* Arms (--arms, --control-arm), the held test snapshot, the ungated baseline, the dumps and the preregistration
  hash.
* Methods (--methods, Task 4.1b): the subset of Plain Erasure, Plain VPE and CPPR that is retrained; CPPR always is.
  Every release is still built, so every graph and every draw is that of a run with all three. Task 4.3 adds the
  Regime-2 baselines of E5 (BASELINES, keeper.cppr.baselines); a list with one of them may leave CPPR out. They take
  no random generator, so they never move a draw.
* preflight_endorser_pools (Task 4.1b) raises an arm's endorser-pool ValueError before any training.
* The base-model cache (--base-cache, Task 4.1c, keeper.cppr.base_cache) leaves every output unchanged, so it is
  not a change of protocol.
* The backbone (--backbone, Task 4.7): seq_scem, the default, is the stored runs'; rating_scem trains every model as the
  rating-level LocalGDTE-SCEM with M-B under the same gate (keeper.cppr.rating_backbone). It needs capped BTP tiers at
  KEEPER's cap and guard (M-B's eps and calibration), --hold-test-snapshot (M-B's levels follow the arm's snapshots),
  the stored methods only, and neither the ungated baseline nor the base-model cache (neither knows the backbone).
"""
import argparse
import hashlib
import math
import os
from dataclasses import dataclass

import numpy as np

from keeper.credibility import tiers as CT
from keeper.credibility.tiers import TIER_WEIGHTS
from keeper.credibility.tiers import legacy_levels
from keeper.laundering import signed as S

TIER_METHODS = ("bfs", "btp")
ENDORSER_SELECTIONS = ("random", "max_trust")
SYBIL_TOPOLOGIES = ("ring", "star")
ARMS = ("direct", "whitewash", "badmouth", "control")
DEFAULT_ARMS = ("direct", "whitewash", "badmouth")         # the stored runs' arms
METHODS = ("plain_erasure", "plain_vpe", "cppr")           # --methods; the stored runs retrain all three
BASELINES = ("author", "singleton", "withdrawal")          # Task 4.3: E5's Regime-2 baselines (keeper.cppr.baselines)
BACKBONES = ("seq_scem", "rating_scem")                    # --backbone (Task 4.7); the first is the stored runs'
METHOD_KEYS = dict(plain_erasure="plain", plain_vpe="vpe", cppr="cppr",   # their names in test_metrics and the dumps
                   **{name: name for name in BASELINES})
FIRST_FAKE_SNAPSHOT = 5         # build_inject_then_erase_graph draws each group's snap_f uniformly from [5, tt)
ENDORSER_MIN_TIER = 2           # random endorsers: base-graph tier >= 2 under the run's tier method (ruling R33)
ATTACK_RNG_ENTROPY = 20261027   # arm_rng: numpy default_rng((ATTACK_RNG_ENTROPY, seed, ARMS.index(arm)))
CAP_TOL = 1e-9                  # slack when flooring the guarantee K g edge_cap_units to a fake count
NEVER_JOINS = np.iinfo(np.int64).max   # legacy_pass: the join snapshot of an account never in D
# common_pool applies ruling R43's criteria (a), (b), (c) and (e) in this order; (d), timing, depends on the group's
# snap_f and comes last (EndorserPicker.candidates).
POOL_CRITERIA = ("bfs_tier", "reference_tier", "not_target", "inflow")


def _positive_finite(value, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _canonical_arms(arms) -> tuple:
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError(f"arms must be distinct names from {ARMS}, got {arms!r}")
    return tuple(a for a in ARMS if a in arms)


def parse_arms(text: str, control: bool = False) -> tuple:
    """The arms of a comma list (--arms), plus control if ``control`` (--control-arm), in the order of ARMS."""
    names = [s.strip() for s in str(text).split(",") if s.strip()]
    if control and "control" not in names:
        names.append("control")
    return _canonical_arms(names)


def _canonical_methods(methods) -> tuple:
    methods, known = tuple(methods), METHODS + BASELINES
    anchored = {"cppr", *BASELINES} & set(methods)
    if len(set(methods)) != len(methods) or any(m not in known for m in methods) or not anchored:
        raise ValueError(f"methods must be distinct names from {known} and include cppr or a baseline, got {methods!r}")
    return tuple(m for m in known if m in methods)


def parse_methods(text: str) -> tuple:
    """The methods of a comma list (--methods), in the order of METHODS then BASELINES; the list must include cppr or
    a Regime-2 baseline (Task 4.3: E5 trains its baselines alone and pairs them with the E3 and E4 runs)."""
    return _canonical_methods(s.strip() for s in str(text).split(",") if s.strip())


def parse_edge_cap(text: str):
    """--btp-edge-cap: 'none' (the uncapped BTP of the approved rule, the ablation) or positive edge_cap_units."""
    if str(text).strip().lower() == "none":
        return None
    try:
        return _positive_finite(text, "--btp-edge-cap")
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None


@dataclass(frozen=True)
class RunProtocol:
    """The Task 3.2 options of one run (see the module docstring); the defaults are the stored runs' protocol."""
    tiers: str = "bfs"
    btp_eps: float | None = None
    btp_edge_cap: float | None = CT.KEEPER_EDGE_CAP_UNITS
    btp_neg_from: str = CT.KEEPER_NEG_FROM
    attack_edges: int = 0
    endorser_select: str = "random"
    sybil_topology: str = "ring"
    arms: tuple = DEFAULT_ARMS
    methods: tuple = METHODS
    hold_test_snapshot: bool = False
    ungated_baseline: bool = False
    dump_targets: str | None = None
    dump_verdicts: str | None = None
    prereg: str | None = None
    pool_eps: float | None = None        # Task 4.1c: eps of the endorser protocol's reference capped calibration
    base_cache: str | None = None        # Task 4.1c: the base-model cache's folder (keeper.cppr.base_cache)
    backbone: str = BACKBONES[0]         # Task 4.7: the models' backbone (keeper.cppr.rating_backbone)

    def __post_init__(self):
        for value, allowed, name in ((self.tiers, TIER_METHODS, "tiers"),
                                     (self.btp_neg_from, CT.NEG_FROM, "btp_neg_from"),
                                     (self.endorser_select, ENDORSER_SELECTIONS, "endorser_select"),
                                     (self.sybil_topology, SYBIL_TOPOLOGIES, "sybil_topology"),
                                     (self.backbone, BACKBONES, "backbone")):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if self.btp_eps is not None:
            object.__setattr__(self, "btp_eps", _positive_finite(self.btp_eps, "btp_eps"))
        elif self.tiers == "btp":
            raise ValueError("--btp-eps is required for --tiers btp")
        if self.btp_eps is not None and self.tiers == "bfs":
            raise ValueError("--btp-eps sets BTP tiers' calibration; a BFS run takes none (the endorser protocol's "
                             "eps is --pool-eps)")
        if self.btp_edge_cap is not None:
            object.__setattr__(self, "btp_edge_cap", _positive_finite(self.btp_edge_cap, "btp_edge_cap"))
        g = self.attack_edges
        if isinstance(g, bool) or int(g) != g or g < 0:
            raise ValueError(f"attack_edges must be a non-negative integer, got {g!r}")
        object.__setattr__(self, "attack_edges", int(g))
        if self.pool_eps is not None:
            object.__setattr__(self, "pool_eps", _positive_finite(self.pool_eps, "pool_eps"))
            if not self.attack_edges:
                raise ValueError("--pool-eps has no use without --attack-edges: it calibrates the endorser protocol")
        elif self.attack_edges:
            raise ValueError("--pool-eps is required with --attack-edges > 0: it sets the reference capped calibration "
                             "of the endorser protocol (rulings R43, R44)")
        if self.tiers == "btp" and self.btp_edge_cap is not None and self.pool_eps not in (None, self.btp_eps):
            raise ValueError(f"--pool-eps ({self.pool_eps}) must equal --btp-eps ({self.btp_eps}) in a capped BTP run")
        object.__setattr__(self, "arms", _canonical_arms(self.arms))
        object.__setattr__(self, "methods", _canonical_methods(self.methods))
        if self.backbone != BACKBONES[0]:
            self._check_rating_backbone()

    def _check_rating_backbone(self) -> None:
        """--backbone rating_scem's requirements (module docstring); ValueError names the first one unmet."""
        needs = (("capped BTP tiers (--tiers btp)", self.tiers == "btp"),
                 (f"KEEPER's cap {CT.KEEPER_EDGE_CAP_UNITS} (--btp-edge-cap)",
                  self.btp_edge_cap == CT.KEEPER_EDGE_CAP_UNITS),
                 (f"KEEPER's guard {CT.KEEPER_NEG_FROM!r} (--btp-neg-from)", self.btp_neg_from == CT.KEEPER_NEG_FROM),
                 ("--hold-test-snapshot", self.hold_test_snapshot),
                 (f"the stored methods only {METHODS}", set(self.methods) <= set(METHODS)),
                 ("no --ungated-baseline", not self.ungated_baseline), ("no --base-cache", self.base_cache is None))
        unmet = [text for text, ok in needs if not ok]
        if unmet:
            raise ValueError(f"--backbone {self.backbone} needs {unmet[0]} (keeper.cppr.rating_backbone)")

    def is_legacy(self) -> bool:
        """True when the computation is the stored runs': legacy tiers, no attack edge, the ring, the default arms,
        all three methods, no held test snapshot, no extra model and the stored backbone. Dumps and the preregistration
        hash only add outputs, the base-model cache changes none, and the BTP options (btp_eps, btp_edge_cap,
        btp_neg_from) change nothing under legacy tiers without attack edges."""
        return (self.tiers == "bfs" and self.attack_edges == 0 and self.sybil_topology == "ring"
                and self.arms == DEFAULT_ARMS and self.methods == METHODS and not self.hold_test_snapshot
                and not self.ungated_baseline and self.backbone == BACKBONES[0])

    @property
    def method_keys(self) -> tuple:
        """The retrained methods under their output names (METHOD_KEYS: plain, vpe, cppr), in the stored order."""
        return tuple(METHOD_KEYS[m] for m in self.methods)

    def calibrate(self, Tr, n_nodes: int):
        """BTP tiers' calibration, frozen on the honest training graph Tr (ruling R32); None under BFS tiers."""
        if self.tiers != "btp":
            return None
        return CT.calibrate_btp(Tr, n_nodes, self.btp_eps, edge_cap_units=self.btp_edge_cap)

    def reference_calibration(self, Tr, n_nodes: int):
        """The endorser protocol's reference capped calibration (eps = pool_eps, edge_cap_units =
        KEEPER_EDGE_CAP_UNITS), frozen on the honest training graph Tr (ruling R32); None without attack edges.
        A capped BTP run at KEEPER's cap computes the same calibration for its tiers, since pool_eps = btp_eps."""
        if not self.attack_edges:
            return None
        return CT.calibrate_btp(Tr, n_nodes, self.pool_eps, edge_cap_units=CT.KEEPER_EDGE_CAP_UNITS)


@dataclass(frozen=True)
class CredibilityTiers:
    """The run's tier function c(u): cred_levels on the base graph and cred_inj on every injected graph.

    bfs: the legacy seeded endorsement of Eq. (14) (keeper.credibility.tiers.legacy_levels, seeded), the stored
         runs' method; it takes no calibration, and it counts every rater's negative ratings whatever neg_from says.
    btp: BTP tiers under ``calibration``, a keeper.credibility.tiers.BTPCalibration frozen on the honest base
         training graph (ruling R32). Fakes then never become seeds and move neither K nor theta0, and under the
         calibration's cap at most floor(K g edge_cap_units) fakes of a region reached by g attack edges reach
         Tier >= 1 (ruling R37). ``neg_from`` is the eligibility guard (ruling R56): "trusted", the default, counts
         only trusted raters' negative ratings, so fakes without attack edges move no honest tier; "all" counts
         every rater's (the ablation).
    """
    method: str = "bfs"
    calibration: object = None
    neg_from: str = CT.KEEPER_NEG_FROM

    def __post_init__(self):
        if self.method not in TIER_METHODS:
            raise ValueError(f"method must be one of {TIER_METHODS}, got {self.method!r}")
        if self.neg_from not in CT.NEG_FROM:
            raise ValueError(f"neg_from must be one of {CT.NEG_FROM}, got {self.neg_from!r}")
        if self.method == "btp" and not isinstance(self.calibration, CT.BTPCalibration):
            raise ValueError("btp tiers need a BTPCalibration frozen on the honest graph (calibrate_btp)")
        if self.method == "bfs" and self.calibration is not None:
            raise ValueError("bfs tiers take no calibration")

    def levels_info(self, Tr, n_nodes: int) -> tuple:
        """(levels, info); info holds the method and, for btp, keeper.credibility.tiers.btp_tiers' info (which
        records neg_from)."""
        if self.method == "bfs":
            return legacy_levels(Tr, n_nodes, seeded=True), dict(method="bfs")
        levels, _, info = CT.btp_tiers(Tr, n_nodes, calibration=self.calibration, neg_from=self.neg_from)
        return levels, dict(method="btp", **info)

    def __call__(self, Tr, n_nodes: int) -> np.ndarray:
        return self.levels_info(Tr, n_nodes)[0]


def arm_rng(seed: int, arm: str) -> np.random.Generator:
    """The arm's own random stream (random endorsers; the control arm's transforms), separate from the stored
    runs' stream, so that an option never moves the snapshots or targets the stored runs drew."""
    return np.random.default_rng((ATTACK_RNG_ENTROPY, int(seed), ARMS.index(arm)))


def legacy_pass(Tr, n_nodes: int, query_rows=()) -> tuple:
    """Replay the single endorsement pass of keeper.credibility.tiers.legacy_levels (seeded) over Tr in stored order.

    Returns (join, sender_in_D). join[v] is the snapshot of the row whose positive rating admitted v to the trusted
    set D: -1 for a seed, NEVER_JOINS for an account that never joins. sender_in_D[k] tells whether the sender of row
    query_rows[k] was in D when the pass read that row. D only grows: negative ratings never remove anyone.
    """
    seeds = CT.seed_set(Tr, n_nodes)
    join = np.full(n_nodes, NEVER_JOINS, dtype=np.int64)
    join[seeds] = -1
    members, query = set(seeds.tolist()), [int(p) for p in np.asarray(query_rows, dtype=np.int64).reshape(-1)]
    wanted, seen = set(query), {}
    rows = zip(np.asarray(Tr.edges).tolist(), S.neg_mask(np.asarray(Tr.labels).reshape(-1, 2)).tolist(),
               np.asarray(Tr.snap).tolist())
    for p, ((u, v), is_neg, snap) in enumerate(rows):
        if p in wanted:
            seen[p] = u in members
        if not is_neg and u in members and v not in members:
            members.add(v)
            join[v] = snap
    return join, np.array([seen[p] for p in query], dtype=bool)


@dataclass(frozen=True, eq=False)
class EndorserInputs:
    """What the endorser protocol reads off the honest base training graph (endorser_inputs; rulings R43, R44).

    bfs_levels: the legacy seeded tiers. reference_levels, reference_inflow: BTP tiers and B(u) under the reference
    capped calibration, whose theta0 is reference_theta0. join: legacy_pass's join snapshots. rank_inflow: the B(u)
    that ranks max-B(u) endorsers (reference_inflow, or the run's own uncapped B(u) under uncapped BTP tiers).
    """
    bfs_levels: np.ndarray
    reference_levels: np.ndarray
    reference_inflow: np.ndarray
    reference_theta0: float
    join: np.ndarray
    rank_inflow: np.ndarray


def endorser_inputs(protocol: RunProtocol, Tr, n_nodes: int, cal, reference):
    """The run's EndorserInputs on its honest base training graph Tr, or None without attack edges. ``cal`` is the run's
    tier calibration (RunProtocol.calibrate) and ``reference`` its reference calibration
    (RunProtocol.reference_calibration). The tier and inflow functions run with their defaults."""
    if not protocol.attack_edges:
        return None
    reference_inflow = CT.endorser_inflow_scores(Tr, n_nodes, reference)
    uncapped = cal is not None and cal.edge_cap_units is None
    return EndorserInputs(bfs_levels=legacy_levels(Tr, n_nodes, seeded=True),
                          reference_levels=CredibilityTiers("btp", reference)(Tr, n_nodes),
                          reference_inflow=reference_inflow, reference_theta0=reference.theta0,
                          join=legacy_pass(Tr, n_nodes)[0],
                          rank_inflow=CT.endorser_inflow_scores(Tr, n_nodes, cal) if uncapped else reference_inflow)


def common_pool(inputs: EndorserInputs, exclude) -> tuple:
    """Ruling R43's random-endorser pool of one arm before timing: the honest accounts u (0 .. n - 1) with
    (a) BFS tier >= 2, (b) reference tier >= 2, (c) u not in ``exclude`` (the arm's targets) and (e) reference
    B(u) >= theta0. Criterion (e) removes the accounts that cannot admit any fake as the only endorser of a region under
    capped BTP: a fake's trust arrives only through the attack edges, and one endorsement from u carries at most B(u)
    over the K steps. With g >= 2, endorsers below theta0 could still reach theta0 together, so (e) is not a necessary
    condition then; its effect is small (about 2 accounts per arm on Bitcoin-Alpha).

    Returns (pool, sizes): the sorted pool, and the pool's size after each criterion, in the order of POOL_CRITERIA,
    as ((criterion, size), ...) after ("honest", n). Criterion (d), timing, is EndorserPicker.candidates."""
    n = len(inputs.bfs_levels)
    masks = (inputs.bfs_levels >= ENDORSER_MIN_TIER, inputs.reference_levels[:n] >= ENDORSER_MIN_TIER,
             ~np.isin(np.arange(n), np.asarray(exclude, dtype=np.int64)),
             inputs.reference_inflow[:n] >= inputs.reference_theta0)
    keep, sizes = np.ones(n, dtype=bool), [("honest", n)]
    for name, mask in zip(POOL_CRITERIA, masks):
        keep &= mask
        sizes.append((name, int(keep.sum())))
    return np.flatnonzero(keep), tuple(sizes)


@dataclass(frozen=True)
class EndorserPicker:
    """A target group's attack-edge endorsers, called as picker(group, snap_f) once the group's snap_f is drawn.

    random: g accounts drawn from ``rng`` uniformly without replacement from ``pool``. With ``join`` (ruling R39, which
        ruling R43 applies under every tier method), the pool shrinks to the accounts already in legacy's trusted set
        D when the single pass reads the group's attack edges (candidates); a group whose timely pool is smaller than
        g raises ValueError. ``criteria`` holds common_pool's sizes when common_pool built the pool (arm_picker).
    max_trust (ruling R44): ``ranking`` holds the honest accounts other than the arm's targets in descending B(u), ties
        to the smaller id; group i (from 0) takes ranking[i g : (i + 1) g].
    """
    selection: str
    g: int
    pool: object = None
    ranking: object = ()
    rng: object = None
    join: object = None
    criteria: tuple = ()

    def candidates(self, snap_f) -> np.ndarray:
        """The accounts a random group drawn at snap_f chooses from."""
        if self.join is None:
            return self.pool
        # R39 condition: joined D in the honest base pass at a snapshot <= snap_f. That suffices: D only grows, and
        # build_inject_then_erase_graph checks (honest_rows_lead) that the honest rows keep their order and that
        # the injected rows of snap_f follow its honest rows, so the pass has read every honest row of a snapshot
        # <= snap_f before the group's attack edges.
        return self.pool[self.join[self.pool] <= snap_f]

    def timely_pool(self, snap_f, where: str) -> np.ndarray:
        """candidates(snap_f); fewer than g raises ruling R39's ValueError, its message prefixed by ``where``."""
        pool = self.candidates(snap_f)
        if len(pool) < self.g:
            raise ValueError(f"{where}: only {len(pool)} candidate endorsers credible at snapshot {snap_f} "
                             f"for {self.g} attack edges (ruling R39)")
        return pool

    def __call__(self, group: int, snap_f) -> list:
        if self.selection == "max_trust":
            ranked = self.ranking[group * self.g:(group + 1) * self.g]
            if len(ranked) < self.g:
                raise ValueError(f"group {group}: only {len(self.ranking)} candidate endorsers for ranks "
                                 f"{group * self.g + 1} to {(group + 1) * self.g} (ruling R44)")
            return [int(u) for u in ranked]
        pool = self.timely_pool(snap_f, f"group {group}")
        return [int(u) for u in self.rng.choice(pool, size=self.g, replace=False)]


def _check_pool(pool, g: int) -> None:
    if g > len(pool):
        raise ValueError(f"only {len(pool)} candidate endorsers for {g} attack edges per group")


def endorser_picker(selection: str, g: int, levels, inflow=None, exclude=(), rng=None, join=None) -> EndorserPicker:
    """An EndorserPicker over the honest accounts 0 .. n - 1 (n = len(levels), or len(inflow) without levels) except
    ``exclude``, the arm's own targets. random draws from the accounts whose ``levels`` reach ENDORSER_MIN_TIER (ruling
    R33's pool; the driver's arms draw from ruling R43's common pool through arm_picker) with a numpy Generator
    ``rng``; ``join`` (legacy join snapshots) applies ruling R39. max_trust needs ``inflow``, the B(u) of
    keeper.credibility.tiers.endorser_inflow_scores, and ranks the accounts by it, ties to the smaller id
    (keeper.analysis.tiers_report.top_accounts), for ruling R44's consecutive groups of g."""
    if selection not in ENDORSER_SELECTIONS:
        raise ValueError(f"selection must be one of {ENDORSER_SELECTIONS}, got {selection!r}")
    g = int(g)
    if g < 0:
        raise ValueError("g must be non-negative")
    if selection == "random":
        if rng is None or levels is None:
            raise ValueError("random endorsers need the base-graph tiers and a numpy Generator")
        ok = np.asarray(levels) >= ENDORSER_MIN_TIER
    else:
        if inflow is None:
            raise ValueError("max_trust endorsers need the inflow scores B(u) (endorser_inflow_scores)")
        inflow = np.asarray(inflow, dtype=np.float64)
        ok = np.ones(len(levels) if levels is not None else len(inflow), dtype=bool)
        if len(inflow) < len(ok):
            raise ValueError(f"{len(inflow)} inflow scores for {len(ok)} honest accounts")
    drop = np.asarray([int(u) for u in exclude], dtype=np.int64)
    ok[drop[(drop >= 0) & (drop < len(ok))]] = False
    pool = np.flatnonzero(ok)
    _check_pool(pool, g)
    if selection == "random":
        return EndorserPicker("random", g, pool=pool, rng=rng, join=join)
    return EndorserPicker("max_trust", g, ranking=pool[np.argsort(-inflow[pool], kind="stable")])


def select_endorsers(selection: str, n_groups: int, g: int, levels, inflow=None, exclude=(), rng=None) -> list:
    """The attack-edge endorsers of each target group, without R39's timing: n_groups lists of g distinct honest
    accounts from endorser_picker (random draws in group order; max_trust gives group i the ranks i g + 1 .. (i + 1) g,
    ruling R44)."""
    if int(n_groups) < 0:
        raise ValueError("n_groups must be non-negative")
    picker = endorser_picker(selection, g, levels, inflow, exclude, rng)
    return [picker(i, None) for i in range(int(n_groups))]


def arm_picker(protocol: RunProtocol, arm: str, seed: int, inputs: EndorserInputs, exclude):
    """The endorser picker of one arm, or None without attack edges; ``inputs`` are the run's EndorserInputs and
    ``exclude`` the arm's targets. random (ruling R43): the arm's common pool (common_pool), restricted at each group's
    snap_f by legacy's join snapshots (ruling R39) under every tier method and drawn from the arm's own stream
    (arm_rng). max_trust (ruling R44): the accounts other than the targets, ranked by inputs.rank_inflow."""
    if not protocol.attack_edges:
        return None
    g = protocol.attack_edges
    if protocol.endorser_select == "max_trust":
        return endorser_picker("max_trust", g, None, inflow=inputs.rank_inflow, exclude=exclude)
    pool, sizes = common_pool(inputs, exclude)
    _check_pool(pool, g)
    return EndorserPicker("random", g, pool=pool, rng=arm_rng(seed, arm), join=inputs.join, criteria=sizes)


def preflight_endorser_pools(protocol: RunProtocol, seed: int, inputs: EndorserInputs, exclude_by_arm: dict,
                             n_groups_by_arm: dict, drawn_from=()) -> None:
    """Raise before any training the ValueError that an arm's endorser picker would raise later (Task 4.1b).

    With g >= 1, each arm of ``exclude_by_arm`` (arm: its targets or, for an arm in ``drawn_from``, the accounts its
    targets will be drawn from) needs, for random endorsers, g accounts in its common pool before timing (ruling R43)
    and in its timely pool at FIRST_FAKE_SNAPSHOT, the smallest timely pool because legacy's trusted set D only grows
    along the pass (ruling R39); for ranked endorsers, g candidates for each of its n_groups_by_arm[arm] groups
    (ruling R44). Excluding a superset of the targets only shrinks the pools, so passing here guarantees every later
    pick. Nothing is drawn: arm_picker's generator is a fresh copy of the arm's stream."""
    if not protocol.attack_edges:
        return
    for arm, exclude in exclude_by_arm.items():
        excluded = "the accounts its targets are drawn from" if arm in drawn_from else "its targets"
        where = f"before training, {arm} arm, pool without {excluded} ({len(exclude)})"
        try:
            picker = arm_picker(protocol, arm, seed, inputs, exclude)
        except ValueError as err:
            raise ValueError(f"{where}: {err}") from err
        if picker.selection == "random":
            picker.timely_pool(FIRST_FAKE_SNAPSHOT, where)
        elif len(picker.ranking) < n_groups_by_arm[arm] * picker.g:
            raise ValueError(f"{where}: only {len(picker.ranking)} candidate endorsers for {n_groups_by_arm[arm]} "
                             f"groups of {picker.g} ranked max-B(u) endorsers (ruling R44)")


def honest_rows_lead(order, snap, n_honest_rows: int) -> bool:
    """The premises of R39's condition for a graph built as [honest rows, injected rows] and stable-sorted by
    snapshot (``order``; ``snap``: the sorted rows' snapshots): the honest rows keep their relative order, and in
    every snapshot each honest row precedes each injected row."""
    order, snap = np.asarray(order), np.asarray(snap)
    injected = order >= n_honest_rows
    if not np.array_equal(order[~injected], np.arange(n_honest_rows)):
        return False
    return not (injected[:-1] & ~injected[1:] & (snap[1:] == snap[:-1])).any()


def _attack_rows(Tr_inj, n_honest: int, n_groups: int, k_fake: int) -> list:
    """Per target group, the stored-order positions of its attack edges in Tr_inj (an honest sender, one of the
    group's fakes as receiver)."""
    src, dst = np.asarray(Tr_inj.edges)[:, 0], np.asarray(Tr_inj.edges)[:, 1]
    return [np.flatnonzero((dst >= n_honest + i * k_fake) & (dst < n_honest + (i + 1) * k_fake) & (src < n_honest))
            for i in range(n_groups)]


def _group_snapshots(Tr_inj, n_honest: int, n_groups: int, k_fake: int) -> np.ndarray:
    """Per target group, in group order, its snap_f: the snapshot of the group's own edges (those its fakes send)."""
    src, snap = np.asarray(Tr_inj.edges)[:, 0], np.asarray(Tr_inj.snap)
    return np.array([int(snap[np.flatnonzero((src >= n_honest + i * k_fake) & (src < n_honest + (i + 1) * k_fake))[0]])
                     for i in range(n_groups)], dtype=np.int64)


def _pool_timing(picker: EndorserPicker, join, snaps) -> dict:
    """A random picker's pool over the groups. Ruling R39's fields: pool_size (min, median and max of the pools the
    groups drew from), pool_size_untimed (the pool before timing), share_removed_by_timing (the mean share of that pool
    not yet in legacy's D at the group's snap_f) and timing_applied (whether the picker restricts by timing, as every
    arm's picker does since ruling R43). pool_criteria (ruling R43): per criterion of common_pool, in order, the pool
    size before and after it and the share of the pool it removes; None for a pool that common_pool did not build."""
    untimed = len(picker.pool)
    timely = (np.array([int((join[picker.pool] <= s).sum()) for s in snaps]) if join is not None
              else np.full(len(snaps), untimed))
    drawn = timely if picker.join is not None else np.full(len(snaps), untimed)
    criteria = [dict(criterion=name, size_before=before, size=size,
                     share_removed=(before - size) / before if before else None)
                for (_, before), (name, size) in zip(picker.criteria, picker.criteria[1:])]
    return dict(pool_size=dict(min=int(drawn.min()), median=float(np.median(drawn)), max=int(drawn.max())),
                pool_size_untimed=int(untimed),
                share_removed_by_timing=float(1.0 - timely.mean() / untimed) if untimed else None,
                timing_applied=picker.join is not None, pool_criteria=criteria or None)


def _ranks(ranking, endorsers) -> list:
    """Each group's endorser ranks in a max_trust picker's ranking, 1 being the largest B(u) (ruling R44)."""
    rank = {int(u): r for r, u in enumerate(ranking, start=1)}
    return [[rank[int(u)] for u in group] for group in endorsers]


def admission_record(cred_inj, Tr_inj, n_honest: int, n_groups: int, k_fake: int, protocol: RunProtocol, tier_fn,
                     picker, join) -> dict:
    """sybil_admission of one injected arm. Each group's snap_f and endorsers are read off the injected training graph
    Tr_inj, and per group each endorser's endorsers_in_D_at_attack tells whether it was in legacy's trusted set D when
    the pass read its attack edge, for either tier method and selection. A random picker adds its pool sizes
    (_pool_timing), and a max_trust picker each group's endorser ranks. snap_f is recorded with or without attack edges
    (Task 4.1b), so that a later run of the same seed can be checked to inject at the same snapshots."""
    snaps = _group_snapshots(Tr_inj, n_honest, n_groups, k_fake)
    if not protocol.attack_edges:
        return sybil_admission(cred_inj, n_honest, n_groups, k_fake, 0, None, tier_fn, snap_f=snaps)
    rows = _attack_rows(Tr_inj, n_honest, n_groups, k_fake)
    _, in_D = legacy_pass(Tr_inj, int(Tr_inj.n), np.concatenate(rows))
    flags = [part.tolist() for part in np.split(in_D, np.cumsum([len(r) for r in rows])[:-1])]
    endorsers = [np.asarray(Tr_inj.edges)[r, 0].tolist() for r in rows]
    selection = None if picker is None else picker.selection
    return sybil_admission(cred_inj, n_honest, n_groups, k_fake, protocol.attack_edges, endorsers, tier_fn,
                           in_D=flags, timing=_pool_timing(picker, join, snaps) if selection == "random" else None,
                           snap_f=snaps, ranks=_ranks(picker.ranking, endorsers) if selection == "max_trust" else None)


def sybil_admission(cred_inj, n_honest: int, n_groups: int, k_fake: int, attack_edges: int, endorsers,
                    tier_fn, in_D=None, timing=None, snap_f=None, ranks=None) -> dict:
    """The fakes' tiers after injection: per group the number in Tier >= 1, the fakes' tier counts, the endorsers,
    each group's snap_f (in group order) and, for BTP tiers with a per-endorsement cap, the guarantee
    floor(K g edge_cap_units) per group (R37). in_D, ranks and timing (admission_record) fill
    endorsers_in_D_at_attack, endorser_ranks (ruling R44) and the pool fields of rulings R39 and R43; None otherwise,
    as is snap_f when not given."""
    levels = np.asarray(cred_inj)[n_honest:n_honest + n_groups * k_fake].reshape(n_groups, k_fake)
    cal = tier_fn.calibration if tier_fn.method == "btp" else None
    guarantee = (math.floor(cal.K * attack_edges * cal.edge_cap_units + CAP_TOL)
                 if cal is not None and cal.edge_cap_units is not None else None)
    admitted = (levels > 0).sum(axis=1)
    pools = timing or dict(pool_size=None, pool_size_untimed=None, share_removed_by_timing=None, timing_applied=None,
                           pool_criteria=None)
    return dict(n_fakes=int(levels.size), n_admitted=int(admitted.sum()), admitted_per_group=admitted.tolist(),
                fake_tier_counts=np.bincount(levels.ravel(), minlength=len(TIER_WEIGHTS)).tolist(),
                endorsers=None if endorsers is None else [[int(u) for u in e] for e in endorsers],
                snap_f=None if snap_f is None else [int(s) for s in snap_f], endorser_ranks=ranks,
                guarantee_per_group=guarantee, endorsers_in_D_at_attack=in_D, **pools)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preregistration(path):
    """dict(path, sha256) of the preregistration file (--prereg), or None without one; a missing file raises."""
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"preregistration file {path} not found")
    return dict(path=str(path), sha256=sha256_file(path))


_CANDIDATES = dict(
    random="ruling R43's common pool: honest accounts other than the arm's targets with BFS and reference tier >= 2, "
           "in legacy's trusted set D when the pass reads the group's attack edges, and reference B(u) >= theta0",
    max_trust="honest accounts other than the arm's targets, ranked by B(u) (ruling R44): the reference B(u) under "
              "BFS and capped tiers, the run's own uncapped B(u) under uncapped tiers")


def _calibration_fields(cal) -> dict:
    """eps, K, theta0, edge_cap_units, edge_cap, n_ref, n_seeds and t_seed of a BTPCalibration; None without one."""
    names = ("eps", "K", "theta0", "edge_cap_units", "edge_cap", "n_ref", "n_seeds", "t_seed")
    return {name: None if cal is None else getattr(cal, name) for name in names}


def config_block(protocol: RunProtocol, gate_norm: str, k_fake: int, cal, prereg, reference=None) -> dict:
    """The output's config block: the tier method and its frozen calibration ``cal``; the attack, with the endorser
    protocol's reference calibration ``reference`` (None without attack edges); arms, methods, test set and
    preregistration. The tier calibration also serves endorser selection when it is the reference one (a capped run
    at KEEPER's cap) or ranks the max-B(u) endorsers of an uncapped run."""
    selects = protocol.attack_edges > 0 and cal is not None and (
        cal == reference or (cal.edge_cap_units is None and protocol.endorser_select == "max_trust"))
    tiers_cfg = dict(method=protocol.tiers, **_calibration_fields(cal),
                     neg_from=protocol.btp_neg_from if protocol.tiers == "btp" else None,  # the guard, ruling R56
                     calibration_used_for=None if cal is None else "tiers and endorser selection" if selects
                     else "tiers")
    attack = dict(attack_edges=protocol.attack_edges, endorser_select=protocol.endorser_select,
                  sybil_topology=protocol.sybil_topology, k_fake=int(k_fake), endorser_min_tier=ENDORSER_MIN_TIER,
                  endorser_candidates=_CANDIDATES[protocol.endorser_select],
                  endorser_assignment="ranked" if protocol.endorser_select == "max_trust" else "common_pool",
                  reference_calibration=None if reference is None else _calibration_fields(reference),
                  random_stream=f"numpy default_rng(({ATTACK_RNG_ENTROPY}, seed, index of the arm in {list(ARMS)}))")
    return dict(tiers=tiers_cfg, attack=attack, arms=list(protocol.arms), methods=list(protocol.methods),
                hold_test_snapshot=protocol.hold_test_snapshot, ungated_baseline=protocol.ungated_baseline,
                gate_norm=gate_norm, preregistration=prereg)
