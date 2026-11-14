"""Adversarial witnesses against honest warners (Task 3.3; the E1 adversarial arm; rulings R34 and R39).

The attack (design-M-B.md, measured limitation). An honest warner W rated a target T negatively at snapshot
t_k; that escrowed warning is the victim record. k fresh Sybils rate W negatively at t_b = t_k + lag. Under
SCEM's stored witness rule each fresh Sybil still carries the floor reliability 0.20, so the hazard they raise
on W revokes part of W's warning and launders T without any honest rater's cooperation. g positive attack
edges from credible honest accounts endorse the Sybils; under M-B they are the only way out of Tier 0.

Timing (ruling R34). Sybils act only after the seed window, so none can become a BTP seed. The attack edges
come at t_a = max(t_k + lag - 1, t_seed + 1) and the bad-mouthing at t_b = t_k + lag >= t_a + 1, so the
strict-prefix tiers of row t_b already include the attack edges. A victim whose t_b would fall at or before
t_seed + 1 is skipped, and so is one whose t_b falls at or beyond the end of the stream (K_snap). Both kinds
are returned, so they can be counted.

Endorsers (ruling R39). A victim's g endorsers are drawn uniformly, without replacement, from the honest
accounts whose causal tier at t_a is at least 2. Those tiers are base_levels[t_a], the clean stream's tiers
under the frozen calibration (ruling R32). W and T are never drawn. The attack edges go round-robin to the
victim's Sybils: endorser j (j = 0, ..., g - 1) endorses Sybil (j mod k) + 1, so with g <= k they reach
Sybils 1..g. Fewer than g eligible accounts raise ValueError.

Sybil ring (option sybil_ring, off by default). An adaptive attacker also links its Sybils by a positive
directed ring s_1 -> s_2 -> ... -> s_k -> s_1 at t_a, the snapshot of the attack edges. Trust from an attack
edge then spreads through the region, and each Sybil receives durable mass, which raises its reliability
kappa: kappa counts durable mass from any rater. k = 1 has no ring (no self-loop).

kappa farming (option kappa_farming, off by default; the E1b variant of protocol Section 2.3). The ring is placed
at t_seed + 1, the earliest snapshot an attacker may use, instead of t_a, so the Sybils collect durable mass, and
with it kappa, for as long as possible before they bad-mouth. The attack edges and the bad-mouthing keep their
snapshots. It needs sybil_ring.

Layout. The injected edges are appended after the original ones, whose ids do not change. They come in victim
order: each victim's g attack edges, then its ring edges, then its k bad-mouthing edges. Every victim gets k
Sybils of its own, with ids from n_nodes upwards. The stream is then no longer sorted by snapshot. The SCEM
engine and keeper.credibility.tiers do not need it to be; np.argsort(snap, kind="stable") restores time order
for a consumer that does.

keeper.scem.property1.property1_bound, re-exported here, bounds per record the direct effect of the Sybils'
witness evidence (design-M-B.md, Property 1). E1 reports it as a descriptive diagnostic (ruling R58), because
indirect paths lie outside it.
"""
from typing import NamedTuple

import numpy as np

from keeper.credibility.tiers import T_SEED
from keeper.scem.property1 import property1_bound  # noqa: F401  (re-export: callers import it from here)

ENDORSER_MIN_LEVEL = 2          # ruling R39: an endorser holds causal tier >= 2 when it endorses


class InjectedStream(NamedTuple):
    """The attacked stream and what was injected. The first six fields are (edges', sign', snap', n_nodes',
    sybil_ids, attack_edge_ids). Row i of the per-victim arrays belongs to victims[i]: its k Sybils, the ids in
    edges' of its g attack edges, its ring edges and its k bad-mouthing edges, its g endorsers and their tiers
    at t_attack[i], the number of eligible endorsers at t_attack[i], and its attack-edge and bad-mouthing
    snapshots. The skipped fields list the record ids that R34 excluded."""
    edges: np.ndarray                   # [E', 2]
    sign: np.ndarray                    # [E']
    snap: np.ndarray                    # [E']
    n_nodes: int
    sybil_ids: np.ndarray               # [n_victims, k]
    attack_edge_ids: np.ndarray         # [n_victims, g]
    ring_edge_ids: np.ndarray           # [n_victims, k] with sybil_ring and k >= 2, else [n_victims, 0]
    badmouth_edge_ids: np.ndarray       # [n_victims, k]
    victims: np.ndarray                 # [n_victims] attacked record ids, in the given order
    endorsers: np.ndarray               # [n_victims, g]
    endorser_levels: np.ndarray         # [n_victims, g] base_levels[t_attack, endorser]
    pool_size: np.ndarray               # [n_victims]
    t_attack: np.ndarray                # [n_victims]
    t_badmouth: np.ndarray              # [n_victims]
    skipped_seed_window: np.ndarray     # record ids with t_k + lag <= t_seed + 1
    skipped_beyond_stream: np.ndarray   # record ids with t_k + lag >= K_snap


def warnings_about(edges, sign, targets) -> np.ndarray:
    """Ids of the warnings about the targets: the negative ratings whose receiver is a target. With the cohort
    targets of keeper.laundering.core.pick_targets, these are the victim records of the E1 adversarial arm."""
    edges = np.asarray(edges).reshape(-1, 2)
    return np.flatnonzero((np.asarray(sign) < 0) & np.isin(edges[:, 1], np.asarray(targets))).astype(np.int64)


def _count(value, name: str, low: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < low:
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return int(value)


def _check_inputs(edges, sign, snap, n_nodes, victims, base_levels, K_snap, rng):
    edges, sign, snap = np.asarray(edges), np.asarray(sign), np.asarray(snap)
    if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges) == len(sign) == len(snap):
        raise ValueError("edges must be [E, 2], with sign and snap of length E")
    if len(edges) and (edges.min() < 0 or edges.max() >= n_nodes):
        raise ValueError(f"edge endpoints must lie in [0, {n_nodes})")
    victims = np.asarray(victims, dtype=np.int64).reshape(-1)
    if len(np.unique(victims)) != len(victims):
        raise ValueError("victim records must be distinct")
    if len(victims) and (victims.min() < 0 or victims.max() >= len(edges)):
        raise ValueError(f"victim records must be edge ids in [0, {len(edges)})")
    if (sign[victims] >= 0).any():
        raise ValueError("every victim record must be a warning (a negative rating)")
    levels = np.asarray(base_levels)
    if levels.ndim != 2 or levels.shape[0] < K_snap or levels.shape[1] != n_nodes:
        raise ValueError(f"base_levels must have shape [T >= {K_snap}, {n_nodes}], got {list(levels.shape)}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator (np.random.default_rng), got {type(rng).__name__}")
    return edges, sign, snap, victims, levels


def inject_adversarial_witnesses(edges, sign, snap, n_nodes: int, victims, k: int, lag: int, g: int, rng,
                                 base_levels, K_snap: int, *, t_seed: int = T_SEED,
                                 sybil_ring: bool = False, kappa_farming: bool = False) -> InjectedStream:
    """Inject k fresh Sybils per victim warning, with g attack edges each and, with sybil_ring, a positive ring
    among them, at t_seed + 1 with kappa_farming (module docstring; rulings R34, R39). Deterministic given rng:
    victims are processed in the given order, and each draws its endorsers in turn."""
    n_nodes, k, lag, g = _count(n_nodes, "n_nodes", 1), _count(k, "k", 1), _count(lag, "lag", 0), _count(g, "g", 0)
    K_snap, t_seed = _count(K_snap, "K_snap", 1), _count(t_seed, "t_seed", 0)
    if kappa_farming and not sybil_ring:
        raise ValueError("kappa_farming places the Sybil ring early; it needs sybil_ring=True")
    edges, sign, snap, victims, levels = _check_inputs(edges, sign, snap, n_nodes, victims, base_levels, K_snap, rng)
    n_ring = k if sybil_ring and k > 1 else 0
    rows, done, seed_window, beyond = [], [], [], []
    per_victim = {key: [] for key in ("sybils", "attack", "ring", "badmouth", "endorsers", "levels", "pool", "t_a",
                                      "t_b")}
    next_id, next_edge = n_nodes, len(edges)
    for rec in victims.tolist():
        warner, target, t_k = int(edges[rec, 0]), int(edges[rec, 1]), int(snap[rec])
        t_b = t_k + lag
        if t_b <= t_seed + 1:
            seed_window.append(rec)
            continue
        if t_b >= K_snap:
            beyond.append(rec)
            continue
        t_a = max(t_k + lag - 1, t_seed + 1)
        eligible = levels[t_a] >= ENDORSER_MIN_LEVEL
        eligible[[warner, target]] = False
        pool = np.flatnonzero(eligible)
        if len(pool) < g:
            raise ValueError(f"victim record {rec}: {len(pool)} honest accounts other than W and T hold tier >= "
                             f"{ENDORSER_MIN_LEVEL} at snapshot {t_a}, fewer than g = {g}")
        chosen = rng.choice(pool, size=g, replace=False) if g else np.zeros(0, dtype=np.int64)
        sybils = np.arange(next_id, next_id + k, dtype=np.int64)
        next_id += k
        rows += [(int(chosen[j]), int(sybils[j % k]), 1, t_a) for j in range(g)]       # round-robin, Sybils 1..g
        t_ring = t_seed + 1 if kappa_farming else t_a
        rows += [(int(sybils[j]), int(sybils[(j + 1) % k]), 1, t_ring) for j in range(n_ring)]
        rows += [(int(s), warner, -1, t_b) for s in sybils]
        ring_start, badmouth_start = next_edge + g, next_edge + g + n_ring
        for key, value in (("sybils", sybils), ("attack", np.arange(next_edge, ring_start)),
                           ("ring", np.arange(ring_start, badmouth_start)),
                           ("badmouth", np.arange(badmouth_start, badmouth_start + k)), ("endorsers", chosen),
                           ("levels", levels[t_a, chosen]), ("pool", len(pool)), ("t_a", t_a), ("t_b", t_b)):
            per_victim[key].append(value)
        next_edge = badmouth_start + k
        done.append(rec)
    new = np.asarray(rows, dtype=np.int64).reshape(-1, 4)

    def block(key, width):          # [n_victims, width]; explicit rows, since width may be 0
        return np.asarray(per_victim[key], dtype=np.int64).reshape(len(done), width)

    def ids(values):
        return np.asarray(values, dtype=np.int64)

    return InjectedStream(
        edges=np.concatenate([edges, new[:, :2].astype(edges.dtype)]),
        sign=np.concatenate([sign, new[:, 2].astype(sign.dtype)]),
        snap=np.concatenate([snap, new[:, 3].astype(snap.dtype)]),
        n_nodes=next_id, sybil_ids=block("sybils", k), attack_edge_ids=block("attack", g),
        ring_edge_ids=block("ring", n_ring), badmouth_edge_ids=block("badmouth", k), victims=ids(done),
        endorsers=block("endorsers", g), endorser_levels=block("levels", g), pool_size=ids(per_victim["pool"]),
        t_attack=ids(per_victim["t_a"]), t_badmouth=ids(per_victim["t_b"]), skipped_seed_window=ids(seed_window),
        skipped_beyond_stream=ids(beyond))
