"""The registered settings of the confirmatory CPPR runs (Task 4.2a), from the protocol in docs/preregistration.md.

This module is the one place that states them. keeper.cluster.cppr_tasks builds the commands of E3 and E4 from them,
and the analyses of Phase 4 import them to assert that every run's recorded config equals them, so the generator and
the analysis cannot disagree. The per-network tables are read-only mappings (types.MappingProxyType); dict(table) makes
a copy that can be serialised. The protocol logs a change after registration in docs/preregistration_deviations.md,
with its date, reason and effect.

Where the protocol states each value:
  NETWORKS                              Section 2.1
  EPS_CAPPED, EPS_UNCAPPED              Section 2.2 and Section 6 (EPS_C, EPS_U)
  K_EXPECTED, N_REF_EXPECTED            Section 2.2 (K = ceil(log2 n_ref); n_ref = active accounts of the honest
                                        calibration graph); recorded by the driver, asserted by the analyses
  EDGE_CAP_UNITS                        Section 2.2 (the capped runs rely on the driver's default, Section 6)
  NEG_FROM                              ruling R56, the credible-negatives guard of BTP's eligibility; it is the
                                        driver's default (--btp-neg-from), so no command passes it
  SEEDS_E3, SEEDS_5                     Section 2.1 (seeds 5-14 for E3; seeds 5-9 for the rest)
  PYTHON, PREREG, EPOCHS                Section 6 (the command templates) and Section 2.1 (150 epochs)
  CAPPED, BFS, UNCAPPED, tier_flags     Section 6 (the flags of the three tier settings)
  E3_ARMS                               Sections 2.5 and 6
  E4_ARMS, E4_METHODS, E4_CELLS, E4_UNCAPPED_CELLS, E4_PAIRED_METHODS, E4_PAIRED_RUN
                                        Sections 2.6 and 6
  E5_ARMS_G0, E5_ARMS_G1, E5_METHODS    Section 2.7 (Task 4.3; keeper.cluster.e5_tasks)
  E2_STREAMS, E2_MODELS, E2_H6_MODELS, E2_H6_STREAMS, SEEDS_E2_H6
                                        Sections 2.1 and 2.4 (the stream draws are keeper.scem.e2_streams')
  E7_BACKBONE, E7_CPPR, E7_WITNESS_STREAMS, E7_MB
                                        Section 2.9 (Task 4.7; keeper.cluster.e7_tasks, keeper.cppr.witness_arm)
"""
from dataclasses import dataclass
from types import MappingProxyType

NETWORKS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")
EPS_CAPPED = MappingProxyType({"bitcoinalpha": 0.02, "bitcoinotc": 0.01, "wiki-rfa": 0.5})
EPS_UNCAPPED = MappingProxyType({"bitcoinalpha": 0.05, "bitcoinotc": 0.05, "wiki-rfa": 0.5})
K_EXPECTED = MappingProxyType({"bitcoinalpha": 12, "bitcoinotc": 13, "wiki-rfa": 14})
N_REF_EXPECTED = MappingProxyType({"bitcoinalpha": 3629, "bitcoinotc": 5684, "wiki-rfa": 10152})
EDGE_CAP_UNITS = 2.0
NEG_FROM = "trusted"
SEEDS_E3 = range(5, 15)
SEEDS_5 = range(5, 10)
PYTHON = "env -u PYTHONPATH -u VIRTUAL_ENV ~/.conda/envs/ps/bin/python"
PREREG = "docs/preregistration.md"
EPOCHS = 150

# The tier settings of the runs, as the labels that file names carry: capped BTP, the legacy tiers, uncapped BTP.
CAPPED, BFS, UNCAPPED = "btpc", "bfs", "btpu"
TIER_LABELS = (CAPPED, BFS, UNCAPPED)

E3_ARMS = "direct,whitewash,badmouth,control"     # Section 2.5; E3 takes the code defaults for the attack edges (g = 0)
E4_ARMS = "whitewash"
E4_METHODS = "cppr"
E4_PAIRED_METHODS = "plain_vpe,cppr"


@dataclass(frozen=True)
class Cell:
    """A cell of E4 (Section 2.6): attack edges per group g, how endorsers are chosen, the region's shape and size s."""
    g: int
    endorsers: str
    topology: str
    s: int


E4_CELLS = (Cell(1, "random", "ring", 3), Cell(1, "random", "ring", 50), Cell(1, "random", "star", 50),
            Cell(1, "max_trust", "star", 3), Cell(1, "max_trust", "star", 50), Cell(1, "max_trust", "ring", 3),
            Cell(2, "random", "ring", 3), Cell(2, "max_trust", "star", 3))
# Section 2.6: "uncapped BTP runs the max-B(u) star s in {3, 50} cells". Read as the two g = 1 cells (18 runs per
# network and seed, 270 in all); the g = 2 max-B(u) star cell runs under BFS and capped BTP only.
E4_UNCAPPED_CELLS = (Cell(1, "max_trust", "star", 3), Cell(1, "max_trust", "star", 50))
E4_PAIRED_RUN = (Cell(1, "random", "ring", 3), CAPPED)    # (cell, tier label) of the run that also trains Plain VPE

# Section 2.7: E5 runs capped BTP on SEEDS_5; at g = 0 the direct and whitewash arms (paired with E3's capped run), at
# g = 1 the whitewash arm in the cell and tier setting of E4_PAIRED_RUN (paired with that run). Only the baselines are
# trained.
E5_ARMS_G0 = "direct,whitewash"
E5_ARMS_G1 = "whitewash"
E5_METHODS = "author,singleton,withdrawal"
# E2: the clean twin and the attack streams at lag 1 and lag 3; the S3R backbone and the SCEM models (learned phi,
# phi fixed at 0.5, without hazard, N = 1). Seeds SEEDS_5 run every stream and model; SEEDS_E2_H6 adds the H6 models
# at lag 1 with their clean twins.
E2_STREAMS = ("clean", "lag1", "lag3")
E2_MODELS = ("s3r", "scem", "scem_phi05", "scem_nohaz", "scem_n1")
E2_H6_MODELS = ("s3r", "scem", "scem_nohaz")
E2_H6_STREAMS = ("clean", "lag1")
SEEDS_E2_H6 = range(5, 15)
# E7 (Section 2.9), capped BTP on SEEDS_5 with the rating-level backbone. The CPPR arms: per attack-edge count g, the
# driver's (arms, methods). g = 0 runs direct laundering (Plain Erasure, CPPR; the run's --methods also retrains Plain
# VPE there) and whitewash; g = 1 runs whitewash with random endorsers, ring, s = 3 (the driver's defaults). The witness
# arm: the clean twin and the attacks at g = 0 and g = 1, each with M-B on and off.
E7_BACKBONE = "rating_scem"
E7_CPPR = MappingProxyType({0: ("direct,whitewash", "plain_erasure,plain_vpe,cppr"),
                            1: ("whitewash", "plain_vpe,cppr")})
E7_WITNESS_STREAMS = MappingProxyType({"clean": None, "g0": 0, "g1": 1})     # stream -> attack edges per warning
E7_MB = ("on", "off")


def tier_flags(label: str, ds: str) -> dict:
    """The driver flags of tier setting `label` on network `ds` (Section 6), as {flag: value} in command order.

    BFS takes no eps. Capped BTP takes its network's capped eps and leaves the cap at the driver's default
    (EDGE_CAP_UNITS). Uncapped BTP takes the uncapped eps and lifts the cap. The eps are read from the tables when
    the function is called. A label outside TIER_LABELS raises ValueError; for a BTP label, a network outside the
    eps tables raises KeyError."""
    if label == BFS:
        return {"--tiers": "bfs"}
    if label == CAPPED:
        return {"--tiers": "btp", "--btp-eps": str(EPS_CAPPED[ds])}
    if label == UNCAPPED:
        return {"--tiers": "btp", "--btp-eps": str(EPS_UNCAPPED[ds]), "--btp-edge-cap": "none"}
    raise ValueError(f"unknown tier setting {label!r}; the labels are {TIER_LABELS}")
