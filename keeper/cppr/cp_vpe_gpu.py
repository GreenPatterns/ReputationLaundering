"""End-to-end PyTorch GPU evaluation of Credibility-Preserving VPE (CP-VPE, the paper's CPPR), the Seeded
Erasure-Proof Credibility Gate, and the Inject-Then-Erase Attack on LocalGDTE with Source-Clocked Escrow Memory (SCEM).

Run from the repository root on one GPU:
    CUDA_VISIBLE_DEVICES=0 ~/.conda/envs/ps/bin/python -m keeper.cppr.cp_vpe_gpu --ds bitcoinalpha --seed 0

One seed trains the gated base model, then the direct-laundering arm (Plain Erasure, Plain VPE, CPPR) and the
inject-then-erase whitewash and bad-mouth arms (Plain VPE, CPPR). The gate defaults to count, CPPR's gate of
Eq. (15); snapshot is the ablation of Table S7 and needs --gate-norm snapshot (ruling G1).

Task 3.2 added the options of keeper.cppr.protocol: BTP tiers under a frozen calibration (--tiers, --btp-eps,
--btp-edge-cap, --btp-neg-from), attack edges into the fakes (--attack-edges, --endorser-select, --sybil-topology),
the arms to run (--arms, --control-arm), --hold-test-snapshot, --ungated-baseline, per-target and per-verdict
dumps (--dump-targets, --dump-verdicts) and the preregistration hash (--prereg). Their defaults compute what
the stored runs did; a changed protocol needs --out, and every output obeys the overwrite guard (ruling R27).

Task 4.1b added --methods (the subset of Plain Erasure, Plain VPE and CPPR to retrain; every release is still
built), a pre-flight check of the random-endorser pools before the base model trains, dump folders made up front,
the result written before its dumps, the control arm's models scored on the bad-mouth victims as well (no extra
training), and each injected arm's per-group snap_f in sybil_admission. The arms' graphs live in
keeper.cppr.arm_graphs. The result and dump files are written atomically (keeper.laundering.core.write_atomic).

Task 4.1c: --pool-eps fixes the endorser protocol's reference capped calibration. Random endorsers come from one pool
for every tier method (ruling R43), and max-B(u) endorsers are ranked per group (ruling R44); see
keeper.cppr.protocol. --base-cache DIR stores the trained base model, or loads a stored one after checking it bit for
bit (keeper.cppr.base_cache). Its key also names the code, the GPU model, torch and the pipeline mode, and by
convention DIR is <repo>/.keeper-cache/base. The result's folder is made before any training.

Task 4.3: --methods also takes E5's Regime-2 baselines author, singleton and withdrawal (keeper.cppr.baselines), with or
without cppr. They take no random generator; every release and draw is still that of a run with the stored methods.
Task 4.7: --backbone rating_scem trains every model as the rating-level LocalGDTE-SCEM with M-B under the same protected
gate (keeper.cppr.rating_backbone); the default seq_scem leaves every training and output as above.

Runbook: if a dump fails after the result was written (the run then stops with the error, and the result's dumps
block names a dump file that does not exist), rerun the seed with --force.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import NamedTuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_scatter import scatter_add  # noqa: E402

from keeper.scem.scem import SCEM  # noqa: E402
from keeper.laundering import core as C  # noqa: E402
from keeper.cppr import linkage as L  # noqa: E402
from keeper.laundering import metrics as M  # noqa: E402
from keeper.laundering import signed as S  # noqa: E402
from keeper.cppr import vpe as V  # noqa: E402

from keeper.credibility.tiers import legacy_levels as compute_credibility_levels  # noqa: E402,F401
from keeper.credibility.tiers import TIER_WEIGHTS as CRED_WEIGHTS  # noqa: E402
from keeper.credibility import tiers as CT  # noqa: E402
from keeper.cppr import base_cache as BC  # noqa: E402
from keeper.cppr import baselines as B  # noqa: E402
from keeper.cppr import dumps as DP  # noqa: E402
from keeper.cppr import rating_backbone as RB  # noqa: E402
from keeper.analysis.provenance import git_state  # noqa: E402
from keeper.cppr.arm_graphs import (build_control_graph, build_inject_then_erase_graph,  # noqa: E402
                                    evaluation_graphs, plain_erasure_graph)
from keeper.cppr.protocol import (ARMS, DEFAULT_ARMS, ENDORSER_MIN_TIER, ENDORSER_SELECTIONS,  # noqa: E402,F401
                                  BACKBONES, BASELINES, FIRST_FAKE_SNAPSHOT, METHODS, NEVER_JOINS, POOL_CRITERIA,
                                  SYBIL_TOPOLOGIES, TIER_METHODS, CredibilityTiers, EndorserInputs, EndorserPicker,
                                  RunProtocol, admission_record, arm_picker, arm_rng, common_pool, config_block,
                                  endorser_inputs, endorser_picker, honest_rows_lead, legacy_pass, parse_arms,
                                  parse_edge_cap, parse_methods, preflight_endorser_pools, preregistration,
                                  select_endorsers, sybil_admission)


@dataclass
class CPVPEConfig:
    granularity: int = 1
    eps: float = float("inf")
    m_min: int = 10
    cred_weights: tuple = CRED_WEIGHTS


def apply_cp_vpe(D: "C.Dataset", G: "C.Graph", tt: int, requests,
                 cred_levels: np.ndarray, cfg: CPVPEConfig,
                 rng: np.random.Generator) -> tuple:
    """Transform the graph via Credibility-Preserving VPE (CP-VPE)."""
    Tr = C.train_split(G, tt)
    erase, to_pool, accounts = V._erased_sets(Tr, requests)
    g = max(1, int(cfg.granularity))
    n_windows = (tt + g - 1) // g
    n_levels = len(cfg.cred_weights)
    n_anon = n_windows * n_levels
    anon_nodes = np.arange(D.n, D.n + n_anon, dtype=np.int64)
    sign = np.where(S.neg_mask(Tr.labels), -1, 1)

    def anon_id(w: int, c_u: int) -> int:
        return int(D.n + w * n_levels + c_u)

    cell_pools = collections.defaultdict(set)
    counts = collections.defaultdict(list)
    for r in np.where(to_pool)[0]:
        w = int(Tr.snap[r]) // g
        u, v, s = int(Tr.edges[r, 0]), int(Tr.edges[r, 1]), int(sign[r])
        c_u = int(cred_levels[u])
        cell_pools[(w, c_u)].add(u)
        counts[(w, c_u, v, s)].append(u)

    merged_pools = {}
    for (w, c_u), members in cell_pools.items():
        if len(members) >= cfg.m_min:
            merged_pools[(w, c_u)] = sorted(members)
        else:
            union_c = set()
            for (w2, c2), m2 in cell_pools.items():
                if c2 == c_u:
                    union_c |= m2
            merged_pools[(w, c_u)] = sorted(union_c)

    new_e, new_lab, new_snap, new_time, pooled = [], [], [], [], []
    pools_by_key = {}
    for (w, c_u, v, s), raters in sorted(counts.items()):
        c = len(raters)
        cell_key = w * n_levels + c_u
        eff_pool = merged_pools[(w, c_u)]
        pools_by_key[cell_key] = eff_pool
        small = len(eff_pool) < cfg.m_min
        if small and np.isfinite(cfg.eps):
            c_rel = max(0, int(round(c + rng.laplace(0.0, 1.0 / cfg.eps))))
        else:
            c_rel = c
        snap_w = min((w + 1) * g - 1, tt - 1)
        t_w = int(G.time[G.snap == snap_w].max()) if (G.snap == snap_w).any() else 0
        a_node = anon_id(w, c_u)
        for _ in range(c_rel):
            new_e.append((a_node, v))
            new_lab.append(s)
            new_snap.append(snap_w)
            new_time.append(t_w)
        pooled.append(dict(
            window=cell_key, raw_window=w, cred_level=c_u,
            target=v, sign=s, raters=sorted(raters), count=c,
            released=c_rel, pool_size=len(eff_pool),
            noised=bool(small and np.isfinite(cfg.eps))
        ))

    keep_full = ~G.mask_of(Tr.rid[erase])
    Gk = G.subset(keep_full)
    if new_e:
        e = np.array(new_e, dtype=np.int64)
        lab = S.one_hot_from_sign(np.array(new_lab))
        rid = np.arange(len(D.edges), len(D.edges) + len(e), dtype=np.int64)
        Gv = C.Graph(
            np.concatenate([Gk.edges, e]),
            np.concatenate([Gk.labels, lab]),
            np.concatenate([Gk.time, np.array(new_time, dtype=np.int64)]),
            np.concatenate([Gk.snap, np.array(new_snap, dtype=np.int64)]),
            D.n + n_anon,
            np.concatenate([Gk.rid, rid]),
        )
    else:
        Gv = C.Graph(Gk.edges, Gk.labels, Gk.time, Gk.snap, D.n + n_anon, Gk.rid)

    Dv = C.Dataset(D.name, Gv.edges, Gv.labels, Gv.time, Gv.n, D.T, D.cfg,
                   meta=dict(D.meta, cp_vpe=True))
    ext_cred = np.zeros(D.n + n_anon, dtype=np.int64)
    ext_cred[:len(cred_levels)] = cred_levels
    for w in range(n_windows):
        for c_u in range(n_levels):
            ext_cred[anon_id(w, c_u)] = c_u

    info = dict(
        granularity=g, eps=cfg.eps, m_min=cfg.m_min,
        n_windows=n_windows, n_levels=n_levels,
        n_erased=int(erase.sum()), n_pooled_verdicts=int(to_pool.sum()),
        n_released_edges=len(new_e), n_accounts=len(accounts),
        pool_sizes={int(k): len(p) for k, p in pools_by_key.items()},
    )
    res = V.VPEResult(Dv, Gv, C.train_split(Gv, tt), anon_nodes, pooled,
                      pools_by_key, accounts, info)
    return res, ext_cred


GATE_NORMS = ("snapshot", "count")


def gate_weights(node_w: torch.Tensor, row: torch.Tensor, col: torch.Tensor,
                 num_nodes: int, norm: str = "snapshot") -> torch.Tensor:
    """Per-edge weights of the credibility gate for one snapshot's in-edges.

    ``count`` (CPPR's gate, Eq. 15 of the paper): a rater's tier weight divided
    by the number of the target's in-edges in this snapshot, so a Tier-0 edge
    keeps weight w_0 even when it is the target's only in-edge.
    ``snapshot`` (the ablation, Supplementary Table S7): a rater's tier weight
    divided by the sum of tier weights over the target's in-edges in this
    snapshot.
    This argument still defaults to ``snapshot``, so existing callers compute
    what they did; --gate-norm and run_single_seed default to count (ruling G1).
    """
    if norm not in GATE_NORMS:
        raise ValueError(f"unknown gate norm {norm!r}; expected one of {GATE_NORMS}")
    ew = node_w[row]
    if norm == "snapshot":
        denom = scatter_add(ew, col, dim=0, dim_size=num_nodes).clamp(min=1e-12)
    else:
        denom = scatter_add(torch.ones_like(ew), col, dim=0, dim_size=num_nodes).clamp(min=1.0)
    return (ew / denom[col]).unsqueeze(-1)


def attach_scem_and_credibility_gate(
    model,
    cred_levels: np.ndarray,
    cred_weights: tuple = CRED_WEIGHTS,
    plain_vpe_anon_nodes: np.ndarray | None = None,
    plain_vpe_pool_weights: dict | None = None,
    gate_norm: str = "snapshot",
):
    """Ensure LocalGDTE uses SCEM as its temporal encoder and install the
    Seeded Erasure-Proof Credibility Gate into StructuralNetwork.
    A model with the G4 self term is refused before any change: the gate would weight each node's self loop by
    the node's own credibility weight node_w[v], an untested combination that no experiment uses.
    """
    if any(getattr(conv, "self_term", False) for conv in model.structural_layer.conv_layers):
        raise ValueError("the credibility gate does not support self_term (upgrade G4): it would weight each self "
                         "loop by its node's credibility weight; build the model with self_term=False")
    dev_struct = model.gpu_structural if getattr(model, "pipeline_mode", False) else model.device
    dev_temp = model.gpu_s3r if getattr(model, "pipeline_mode", False) else model.device

    if not isinstance(getattr(model, "s3r", None), SCEM):
        scem_mod = SCEM(
            input_dim=model.input_dim,
            n_heads=model.num_heads,
            max_seq_len=int(getattr(model.args, "max_snapshots_inaClient", 0) or model.args.train_time_slots),
            dropout=model.dropout,
        ).to(dev_temp)
        model.scem = scem_mod
        model.s3r = scem_mod

    w_table = torch.tensor(cred_weights, dtype=torch.float32, device=dev_struct)
    c_tensor = torch.as_tensor(cred_levels, dtype=torch.long, device=dev_struct)
    node_w = w_table[c_tensor].clone()

    if plain_vpe_anon_nodes is not None and plain_vpe_pool_weights is not None:
        for w_idx, a_id in enumerate(plain_vpe_anon_nodes):
            if int(w_idx) in plain_vpe_pool_weights and int(a_id) < len(node_w):
                node_w[int(a_id)] = float(plain_vpe_pool_weights[int(w_idx)])

    with torch.no_grad():
        X_raw = model.structural_layer.X_raw.to(dev_struct)
        tier_proto = torch.zeros((len(cred_weights), X_raw.shape[1]), dtype=X_raw.dtype, device=dev_struct)
        for c_id in range(len(cred_weights)):
            mask = (c_tensor == c_id)
            if mask.any():
                tier_proto[c_id] = X_raw[mask].mean(dim=0)
        model.structural_layer.X_raw.copy_(tier_proto[c_tensor])

    for conv in model.structural_layer.conv_layers:
        def make_incoming(layer_ref):
            def gated_incoming_edges(feature_mu, label_mu, edge_index, external_trust=None):
                row, col = edge_index
                num_nodes = feature_mu.size(0)
                norm_w = gate_weights(node_w, row, col, num_nodes, gate_norm)
                opinion_mu = scatter_add(label_mu * norm_w, col, dim=0, dim_size=num_nodes)
                out_mu = scatter_add(feature_mu[row] * norm_w, col, dim=0, dim_size=num_nodes)
                return out_mu, opinion_mu
            return gated_incoming_edges

        conv.incoming_edges = make_incoming(conv)
    return model


def train_gated_model(D: "C.Dataset", Tr: "C.Graph", seed: int, tt: int,
                      cred_levels: np.ndarray, epochs: int = 150,
                      plain_vpe_anon_nodes: np.ndarray | None = None,
                      plain_vpe_pool_weights: dict | None = None,
                      gate_norm: str = "snapshot"):
    """Build and train LocalGDTE-SCEM with the Seeded Erasure-Proof Credibility Gate."""
    m = C.make_model(D, Tr, seed, tt)
    attach_scem_and_credibility_gate(
        m, cred_levels, CRED_WEIGHTS,
        plain_vpe_anon_nodes=plain_vpe_anon_nodes,
        plain_vpe_pool_weights=plain_vpe_pool_weights,
        gate_norm=gate_norm,
    )
    return C.train(m, Tr, tt, epochs=epochs)


class ArmModel(NamedTuple):
    """A model retrained on a Plain VPE or CPPR release, with its tiers, embedding and targets' reputations."""
    res: object
    cred: np.ndarray
    model: object
    z: object
    r_after: np.ndarray


def _retention_arm(D_a, G_a, cred_a, reqs_a, targets, panel, tt, g, seed, epochs, gate_norm, rng,
                   methods=("vpe", "cppr"), train=None) -> dict:
    """Plain VPE, then CPPR, then the baselines in ``methods`` (keeper.cppr.baselines), on one arm's graph and
    requests, retrained as stored: dict(vpe=ArmModel, cppr=..., ...). Both stored releases are built; a method outside
    ``methods`` (--methods) is not retrained and has no entry. At eps = inf neither transform draws from rng (see
    run_single_seed), and the baselines take no generator. ``train`` is the backbone's trainer (train_gated_model)."""
    train = train or train_gated_model
    w_arr = np.asarray(CRED_WEIGHTS, dtype=np.float64)
    res_vpe = V.apply_vpe(D_a, G_a, tt, reqs_a, V.VPEConfig(g, float("inf"), 10), rng)
    models = {}
    if "vpe" in methods:
        pool_w = {
            int(w_idx): float(np.mean([w_arr[cred_a[u]] for u in p_members]))
            for w_idx, p_members in res_vpe.pools.items() if len(p_members) > 0
        }
        cred_vpe = np.concatenate([cred_a, np.zeros(len(res_vpe.anon_nodes), dtype=np.int64)])
        m_vpe = train(res_vpe.D, res_vpe.Tr, seed, tt, cred_vpe, epochs=epochs, gate_norm=gate_norm,
                      plain_vpe_anon_nodes=res_vpe.anon_nodes,
                      plain_vpe_pool_weights=pool_w)
        z_vpe = C.embed(m_vpe, res_vpe.Tr, tt)
        models["vpe"] = ArmModel(res_vpe, cred_vpe, m_vpe, z_vpe, panel.reputation(m_vpe, z_vpe, np.array(targets)))
    res_cp, cred_cp = apply_cp_vpe(D_a, G_a, tt, reqs_a, cred_a, CPVPEConfig(g, float("inf"), 10), rng)
    trained = {"cppr": B.Release(res_cp, cred_cp)} if "cppr" in methods else {}
    for name, rel in dict(trained, **B.releases(D_a, G_a, tt, reqs_a, cred_a, res_cp, methods)).items():
        m = (rel.train or train)(rel.res.D, rel.res.Tr, seed, tt, rel.cred, epochs=epochs, gate_norm=gate_norm)
        z = C.embed(m, rel.res.Tr, tt)
        models[name] = ArmModel(rel.res, rel.cred, m, z, panel.reputation(m, z, np.array(targets)))
    return models


def _retention_block(models: dict, r0, vpe_key: str, cp_key: str, ratio=None) -> dict:
    """The result block of a whitewash, bad-mouth or control arm: the mean of r_after - r0 of each retrained method,
    Plain VPE's under vpe_key and CPPR's under cp_key, and ratio(vpe, cppr) as suppression_ratio when both were; then
    each baseline's under vpe_key with plain_vpe replaced by its name (gain_author, drop_singleton, ...)."""
    mean = {name: float(np.mean(am.r_after - np.array(r0))) for name, am in models.items()}
    block = {key: mean[name] for name, key in (("vpe", vpe_key), ("cppr", cp_key)) if name in mean}
    if ratio is not None and {"vpe", "cppr"} <= set(mean):
        block["suppression_ratio"] = ratio(mean["vpe"], mean["cppr"])
    return dict(block, **{vpe_key.replace("plain_vpe", name): mean[name] for name in BASELINES if name in mean})


def _fmt(x, spec: str) -> str:
    return "None" if x is None else format(x, spec)


@dataclass
class _RunContext:
    """What the arms of one seed share; held, target_rows and verdict_rows collect the optional outputs."""
    D: object
    G: object
    Tr: object
    tt: int
    panel: object
    seed: int
    epochs: int
    gate_norm: str
    g: int
    hold: bool
    methods: tuple                  # the retrained methods under their output names (RunProtocol.method_keys)
    held: dict
    target_rows: list | None
    verdict_rows: list | None
    train: object = None            # the base graph's trainer (train_gated_model, or the backbone's: RB.Backbone)


class _DirectFit(NamedTuple):
    """A method of the direct arm, retrained: its model, test metrics, the edge and account targets' reputations,
    the graph it trained on, and its release, tiers and linkage (None for Plain Erasure)."""
    model: object
    tm: dict
    r: tuple
    Tr: object
    res: object
    cred: object
    linkage: object


# The direct_laundering block in the stored order: each key, the methods it needs and its value from the per-method
# means s[method] = dict(edge, account, mcc, linkage). A key that needs a method left out by --methods is absent.
_DIRECT_FIELDS = (
    ("gain_plain_edge", ("plain",), lambda s: s["plain"]["edge"]),
    ("gain_vpe_edge", ("vpe",), lambda s: s["vpe"]["edge"]),
    ("gain_cp_vpe_edge", ("cppr",), lambda s: s["cppr"]["edge"]),
    ("reduction_plain_vpe", ("plain", "vpe"),
     lambda s: None if not s["plain"]["edge"] else 1.0 - s["vpe"]["edge"] / s["plain"]["edge"]),
    ("reduction_cp_vpe", ("plain", "cppr"),
     lambda s: None if not s["plain"]["edge"] else 1.0 - s["cppr"]["edge"] / s["plain"]["edge"]),
    ("gain_plain_account", ("plain",), lambda s: s["plain"]["account"]),
    ("gain_vpe_account", ("vpe",), lambda s: s["vpe"]["account"]),
    ("gain_cp_vpe_account", ("cppr",), lambda s: s["cppr"]["account"]),
    ("t1_mcc_plain", ("plain",), lambda s: s["plain"]["mcc"]),
    ("t1_mcc_vpe", ("vpe",), lambda s: s["vpe"]["mcc"]),
    ("t1_mcc_cp_vpe", ("cppr",), lambda s: s["cppr"]["mcc"]),
    ("d_t1_mcc_cp_vs_plain", ("plain", "cppr"), lambda s: s["cppr"]["mcc"] - s["plain"]["mcc"]),
    ("linkage_plain_vpe", ("vpe",), lambda s: s["vpe"]["linkage"]),
    ("linkage_cp_vpe", ("cppr",), lambda s: s["cppr"]["linkage"]),
)


def _direct_arm(ctx: _RunContext, reqs, cred_levels, edge_t, acc_t, r0_cohort, rng) -> dict:
    """The stored direct-laundering arm (Plain Erasure, Plain VPE, CPPR) in the stored order, then the baselines in
    ctx.methods (keeper.cppr.baselines): its result block. Each stored release is built; a method outside ctx.methods
    (--methods) is not retrained, and its fields and rows are absent. Retain with author is the base model, not
    retrained (B.AUTHOR_DIRECT_NOTE)."""
    D, G, tt = ctx.D, ctx.G, ctx.tt
    r0 = (np.array([r0_cohort[v] for v in edge_t]), np.array([r0_cohort[v] for v in acc_t]))
    fits = {}

    def fit(D_m, G_m, Tr_m, cred, res=None, train=None, **pools) -> _DirectFit:
        m = (train or ctx.train)(D_m, Tr_m, ctx.seed, tt, cred, epochs=ctx.epochs, gate_norm=ctx.gate_norm, **pools)
        z = C.embed(m, Tr_m, tt)
        tm = C.test_metrics(m, z, *evaluation_graphs(ctx.hold, D, G, D_m, G_m))
        r = (ctx.panel.reputation(m, z, np.array(edge_t)), ctx.panel.reputation(m, z, np.array(acc_t)))
        return _DirectFit(m, tm, r, Tr_m, res, cred, None if res is None else L.linkage_attack(res, m, tt))

    erase_rids, _ = V.plain_erasure(D, G, tt, reqs)
    Dp, Gp, Trp = plain_erasure_graph(D, G, tt, erase_rids, ctx.hold)
    if "plain" in ctx.methods:
        fits["plain"] = fit(Dp, Gp, Trp, cred_levels)
    res_vpe = V.apply_vpe(D, G, tt, reqs, V.VPEConfig(ctx.g, float("inf"), 10), rng)
    if "vpe" in ctx.methods:
        w_arr = np.asarray(CRED_WEIGHTS, dtype=np.float64)
        pool_w_vpe = {
            int(w_idx): float(np.mean([w_arr[cred_levels[u]] for u in p_members]))
            for w_idx, p_members in res_vpe.pools.items() if len(p_members) > 0
        }
        cred_vpe = np.concatenate([cred_levels, np.zeros(len(res_vpe.anon_nodes), dtype=np.int64)])
        fits["vpe"] = fit(res_vpe.D, res_vpe.G, res_vpe.Tr, cred_vpe, res_vpe, plain_vpe_anon_nodes=res_vpe.anon_nodes,
                          plain_vpe_pool_weights=pool_w_vpe)
    res_cp, cred_cp = apply_cp_vpe(D, G, tt, reqs, cred_levels, CPVPEConfig(ctx.g, float("inf"), 10), rng)
    if "cppr" in ctx.methods:
        fits["cppr"] = fit(res_cp.D, res_cp.G, res_cp.Tr, cred_cp, res_cp)
    for name, rel in B.releases(D, G, tt, reqs, cred_levels, res_cp, ctx.methods, erased=(Dp, Gp)).items():
        fits[name] = (_DirectFit(None, ctx.held["base"], r0, ctx.Tr, None, cred_levels, None) if name == "author" else
                      fit(rel.res.D, rel.res.G, rel.res.Tr, rel.cred, rel.res if rel.res.pooled else None, rel.train))

    if ctx.hold:
        ctx.held["direct"] = {name: f.tm for name, f in fits.items()}
    if ctx.target_rows is not None:
        for name, f in fits.items():
            ctx.target_rows += DP.target_rows("direct", name, "edge", edge_t, r0[0], f.r[0], ctx.Tr, f.Tr)
            ctx.target_rows += DP.target_rows("direct", name, "account", acc_t, r0[1], f.r[1], ctx.Tr, f.Tr)
    if ctx.verdict_rows is not None:
        for name, f in fits.items():
            if f.res is not None:
                ctx.verdict_rows += DP.verdict_rows("direct", name, f.res, f.model, tt, f.cred)
    s = {name: dict(edge=M.mean_or_none((f.r[0] - r0[0]).tolist()), account=M.mean_or_none((f.r[1] - r0[1]).tolist()),
                    mcc=f.tm["t1"]["mcc"], linkage=f.linkage) for name, f in fits.items()}
    return {key: value(s) for key, need, value in _DIRECT_FIELDS if set(need) <= set(s)} | B.direct_fields(s)


def _finish_arm(ctx: _RunContext, arm: str, kind: str, targets, r0, before, models: dict) -> None:
    """Held test metrics (on the original test snapshot) and dump rows of a whitewash, bad-mouth or control arm;
    ``before`` is the arm's training graph before erasure and ``models`` the retrained methods' ArmModels from
    _retention_arm: cppr always, and vpe unless --methods leaves Plain VPE out."""
    if ctx.hold:
        ctx.held[arm] = {name: C.test_metrics(am.model, am.z, *evaluation_graphs(True, ctx.D, ctx.G, am.res.D,
                                                                                  am.res.G))
                         for name, am in models.items()}
    for name, am in models.items():
        if ctx.target_rows is not None:
            ctx.target_rows += DP.target_rows(arm, name, kind, targets, r0, am.r_after, before, am.res.Tr)
        if ctx.verdict_rows is not None:
            ctx.verdict_rows += DP.verdict_rows(arm, name, am.res, am.model, ctx.tt, am.cred)


def run_single_seed(ds: str, seed: int, epochs: int = 150, rho: float = 0.02,
                    g: int = 1, k_fake: int = 3, n_edge: int = 15, n_acc: int = 5,
                    out_name: str | None = None, gate_norm: str = "count", force: bool = False,
                    protocol: RunProtocol | None = None):
    """One seed of the CPPR evaluation; ``g`` is the pool-window granularity and ``protocol`` the Task 3.2 options
    (None: the stored runs' protocol). Writes the result to results/laundering/ (or out_name) and returns it."""
    protocol = RunProtocol() if protocol is None else protocol
    T0 = time.time()
    if out_name is None and not protocol.is_legacy():
        raise ValueError("a protocol other than the stored runs' needs an explicit output name (--out NAME)")
    if k_fake < 1:
        raise ValueError(f"k_fake must be at least 1, got {k_fake}")
    result_name = out_name or (f"cp_vpe_{ds}_s{seed}.json" if gate_norm == "snapshot"
                               else f"cpgate_{gate_norm}_{ds}_s{seed}.json")
    result_file = C.result_path(result_name, force)   # refuse an existing result before the training, not after it
    dump_files = DP.dump_paths(result_file, dict(targets=protocol.dump_targets, verdicts=protocol.dump_verdicts), force)
    prereg = preregistration(protocol.prereg)
    git = git_state() if protocol.base_cache else None      # the cache key names the code by its commit (fix round 1)
    if git is not None and git["git_commit"] is None:
        raise ValueError("--base-cache needs a git checkout: the cache key names the code by its commit")
    # Made now, so that a folder that cannot be made stops the run before training: the dump folders (Task 4.1b), the
    # result's and the base-model cache's (Task 4.1c).
    for folder in [os.path.dirname(result_file), *[os.path.dirname(p) for p in dump_files.values()],
                   *([protocol.base_cache] if protocol.base_cache else [])]:
        os.makedirs(folder, exist_ok=True)
    C.set_deterministic(True)
    C.require_gpu()
    D = C.load(ds)
    mem = C.apply_memory_profile(D)
    G = C.full_graph(D)
    tt = D.tt
    Tr = C.train_split(G, tt)
    panel = C.Panel(Tr, n_probe=256, seed=1234)
    cal = protocol.calibrate(Tr, D.n)                    # BTP tiers', frozen on the honest training graph (ruling R32)
    reference = protocol.reference_calibration(Tr, D.n)  # the endorser protocol's, likewise (rulings R43, R44)
    tier_fn = CredibilityTiers("btp", cal, protocol.btp_neg_from) if protocol.tiers == "btp" else CredibilityTiers("bfs")
    cred_levels = tier_fn(Tr, D.n)
    bb = RB.Backbone(protocol, Tr, tt)      # Task 4.7: its trainers are None under the default backbone seq_scem
    inputs = endorser_inputs(protocol, Tr, D.n, cal, reference)     # None without attack edges
    join = None if inputs is None else inputs.join

    cohort, neg, pos = C.pick_targets(Tr, max_neg=3, strict=True)
    trusted_cohort = np.array(sorted([
        v for v, p_rows in pos.items() if len(p_rows) >= 3 and len(neg.get(v, [])) == 0
    ]), dtype=np.int64)
    # The workload uses its own generators and no model, so it is computed before the base model (Task 4.1b).
    rng = np.random.default_rng(3000 + seed)
    honest_rng = np.random.default_rng(int(1e6 * rho) + 17 * seed)
    reqs, edge_t, acc_t = V.workload(Tr, cohort, neg, n_edge, n_acc, rho,
                                     3000 + seed, honest_rng) if hasattr(V, "workload") else (None, None, None)
    if reqs is None:
        from keeper.cppr.vpe_frontier import workload
        reqs, edge_t, acc_t = workload(Tr, cohort, neg, n_edge, n_acc, rho, 3000 + seed, honest_rng)
    # The endorser pickers' errors, raised before any training (Task 4.1b). Both injected graphs are built in every
    # run, so both arms are checked. The bad-mouth targets are drawn from rng only after the whitewash graph, so that
    # arm is checked without the whole trusted cohort they come from, a lower bound on its pool.
    preflight_endorser_pools(protocol, seed, inputs, dict(whitewash=edge_t, badmouth=trusted_cohort),
                             dict(whitewash=len(edge_t), badmouth=min(len(edge_t), len(trusted_cohort))),
                             drawn_from=("badmouth",))

    # With --base-cache, a stored base model of the run's key is built through train_gated_model with 0 epochs and
    # loaded; its outputs, recomputed here, must equal the stored ones bit for bit (Task 4.1c).
    cache = None if protocol.base_cache is None else BC.BaseCache.for_run(
        protocol.base_cache, BC.identity(ds, seed, epochs, gate_norm, protocol.tiers, cal, git["git_commit"]),
        cred_levels, Tr, git["git_dirty"])
    entry = None if cache is None else cache.load()
    train0 = bb.trainer(G, "base") or train_gated_model          # the base model's and the direct arm's
    m0 = train0(D, Tr, seed, tt, cred_levels, epochs=epochs if entry is None else 0, gate_norm=gate_norm)
    if entry is not None:
        m0.load_state_dict(entry["state_dict"])
    z0 = C.embed(m0, Tr, tt)
    rep_cohort, rep_trusted = panel.reputation(m0, z0, cohort), panel.reputation(m0, z0, trusted_cohort)
    r0_cohort = dict(zip(cohort.tolist(), rep_cohort.tolist()))
    r0_trusted = dict(zip(trusted_cohort.tolist(), rep_trusted.tolist()))
    base_tm = C.test_metrics(m0, z0, D, G)
    used = None if cache is None else cache.check_or_store(entry, m0, z0, rep_cohort, rep_trusted, base_tm)

    honest_only_reqs = [q for q in reqs if q.tag == "honest"]
    ctx = _RunContext(D, G, Tr, tt, panel, seed, epochs, gate_norm, g, protocol.hold_test_snapshot,
                      protocol.method_keys, dict(base=base_tm), [] if "targets" in dump_files else None,
                      [] if "verdicts" in dump_files else None, train0)
    arms, blocks, admission = protocol.arms, {}, {}
    if "direct" in arms:
        blocks["direct_laundering"] = _direct_arm(ctx, reqs, cred_levels, edge_t, acc_t, r0_cohort, rng)

    # Both injected graphs are built in every run, so the stored runs' draws from rng (each group's snap_f, then
    # the bad-mouth targets) do not depend on the arms that run. Nor on --methods: every release is built, and
    # Plain VPE and CPPR draw from rng only at a finite eps (the Laplace noise of small pools), while this driver
    # passes eps = inf. Random endorsers draw from their arm's own stream (arm_rng; test I22).
    r0_edge = [r0_cohort[v] for v in edge_t]
    picker_ww = arm_picker(protocol, "whitewash", seed, inputs, edge_t)
    D_ww, G_ww, cred_ww, reqs_ww = build_inject_then_erase_graph(
        D, G, tt, cred_levels, edge_t, +1, k_fake, honest_only_reqs, rng, sybil_ring=True, seeded=True,
        endorsers=picker_ww, topology=protocol.sybil_topology, tier_fn=tier_fn)
    if "whitewash" in arms:
        ww = _retention_arm(D_ww, G_ww, cred_ww, reqs_ww, edge_t, panel, tt, g, seed, epochs, gate_norm, rng,
                            ctx.methods, bb.trainer(G_ww, "whitewash"))
        blocks["inject_then_erase_whitewash"] = _retention_block(ww, r0_edge, "gain_plain_vpe", "gain_cp_vpe",
                                                                 lambda vpe, cp: 1.0 - (cp / max(vpe, 1e-9)))
        Tr_ww = C.train_split(G_ww, tt)
        admission["whitewash"] = admission_record(cred_ww, Tr_ww, D.n, len(edge_t), k_fake, protocol, tier_fn,
                                                  picker_ww, join)
        _finish_arm(ctx, "whitewash", "edge", edge_t, r0_edge, Tr_ww, ww)

    bm_targets = rng.choice(trusted_cohort, size=min(len(edge_t), len(trusted_cohort)), replace=False).tolist()
    r0_bm = [r0_trusted[v] for v in bm_targets]
    picker_bm = arm_picker(protocol, "badmouth", seed, inputs, bm_targets)
    D_bm, G_bm, cred_bm, reqs_bm = build_inject_then_erase_graph(
        D, G, tt, cred_levels, bm_targets, -1, k_fake, honest_only_reqs, rng, sybil_ring=False, seeded=True,
        endorsers=picker_bm, topology=protocol.sybil_topology, tier_fn=tier_fn)
    if "badmouth" in arms:
        bm = _retention_arm(D_bm, G_bm, cred_bm, reqs_bm, bm_targets, panel, tt, g, seed, epochs, gate_norm, rng,
                            ctx.methods, bb.trainer(G_bm, "badmouth"))
        blocks["inject_then_erase_badmouth"] = _retention_block(bm, r0_bm, "drop_plain_vpe", "drop_cp_vpe",
                                                                lambda vpe, cp: 1.0 - (abs(cp) / max(abs(vpe), 1e-9)))
        Tr_bm = C.train_split(G_bm, tt)
        admission["badmouth"] = admission_record(cred_bm, Tr_bm, D.n, len(bm_targets), k_fake, protocol, tier_fn,
                                                 picker_bm, join)
        _finish_arm(ctx, "badmouth", "trusted", bm_targets, r0_bm, Tr_bm, bm)

    if "control" in arms:
        D_c, G_c, cred_c, reqs_c = build_control_graph(D, G, cred_levels, honest_only_reqs)
        ctl = _retention_arm(D_c, G_c, cred_c, reqs_c, edge_t, panel, tt, g, seed, epochs, gate_norm,
                             arm_rng(seed, "control"), ctx.methods, bb.trainer(G_c, "control"))
        # The same models scored on the bad-mouth victims too, without training (Task 4.1b): H2b's control-adjusted
        # endpoint (VPE_bm - VPE_ctl) - (CPPR_bm - CPPR_ctl) per victim. Scoring draws no random number.
        ctl_bm = {name: am._replace(r_after=panel.reputation(am.model, am.z, np.array(bm_targets)))
                  for name, am in ctl.items()}
        blocks["inject_then_erase_control"] = dict(
            _retention_block(ctl, r0_edge, "gain_plain_vpe", "gain_cp_vpe"),
            **_retention_block(ctl_bm, r0_bm, "gain_plain_vpe_badmouth", "gain_cp_vpe_badmouth"),
            n_fakes=0, n_requests=len(reqs_c))
        _finish_arm(ctx, "control", "edge", edge_t, r0_edge, Tr, ctl)
        if ctx.target_rows is not None:
            for name, am in ctl_bm.items():
                ctx.target_rows += DP.target_rows("control", name, "trusted", bm_targets, r0_bm, am.r_after, Tr,
                                                  am.res.Tr)

    ungated = None
    if protocol.ungated_baseline:
        m_u = C.train(C.make_model(D, Tr, seed, tt), Tr, tt, epochs=epochs)
        ungated = C.test_metrics(m_u, C.embed(m_u, Tr, tt), D, G)

    out = dict(
        ds=ds, seed=seed, epochs=epochs, rho=rho, granularity=g, k_fake=k_fake,
        gate_norm=gate_norm,
        temporal_encoder="SCEM",
        memory_profile=mem, base_t1=base_tm["t1"],
        **blocks,
        config=dict(config_block(protocol, gate_norm, k_fake, cal, prereg, reference), **bb.config()),
        sybil_admission=admission,
    )
    if protocol.hold_test_snapshot:
        out["test_metrics"] = ctx.held
    if ungated is not None:
        out["ungated_baseline"] = ungated
    if cache is not None:                            # with the meta of the base model used (GPU, torch, created, commit)
        out["base_model"] = dict(from_cache=entry is not None, key=cache.key, **{k: used[k] for k in BC.RESULT_FIELDS})
    prov = dict(C.provenance(D), **git_state())      # core.dump's provenance plus the code commit
    tables = dict(targets=(DP.TARGET_COLUMNS, ctx.target_rows), verdicts=(DP.VERDICT_COLUMNS, ctx.verdict_rows))
    if dump_files:
        out["dumps"] = {kind: dict(path=p, n_rows=len(tables[kind][1])) for kind, p in dump_files.items()}
    out["runtime_s"] = round(time.time() - T0, 1)
    path = C.dump(dict(out, provenance=prov), result_name, D, force=force)    # the result first: a dump that fails
    for kind, dump_path in dump_files.items():                                # later cannot cost it (Task 4.1b)
        columns, rows = tables[kind]
        C.dump(DP.dump_table(columns, rows, dict(result=os.path.basename(result_file), config=out["config"],
                                                 provenance=prov)), dump_path, D, force=force)
    print(f"[{ds} s={seed} SCEM] " + " | ".join(_summary(blocks)) + f" ({out['runtime_s']}s)")
    print(f"wrote {path}")
    return out


def _summary(blocks: dict) -> list:
    """The printed one-line summary of the arms that ran (None for a method that --methods left out)."""
    parts = []
    if "direct_laundering" in blocks:
        b = blocks["direct_laundering"]
        parts.append(f"Direct laundering: plain={_fmt(b.get('gain_plain_edge'), '.4f')} -> "
                     f"VPE={_fmt(b.get('gain_vpe_edge'), '.4f')} -> CP-VPE={_fmt(b.get('gain_cp_vpe_edge'), '.4f')}")
    for key, label, a, b in (("inject_then_erase_whitewash", "Inject-Erase WW", "gain_plain_vpe", "gain_cp_vpe"),
                             ("inject_then_erase_badmouth", "Inject-Erase BM", "drop_plain_vpe", "drop_cp_vpe"),
                             ("inject_then_erase_control", "Control", "gain_plain_vpe", "gain_cp_vpe")):
        if key in blocks:
            parts.append(f"{label}: VPE={_fmt(blocks[key].get(a), '.4f')} -> CP-VPE={_fmt(blocks[key].get(b), '.4f')}")
    if "linkage_cp_vpe" in blocks.get("direct_laundering", {}):
        parts.append(f"CP-VPE AUC={_fmt(blocks['direct_laundering']['linkage_cp_vpe']['auc']['combined'], '.3f')}")
    named = [f"{k}={v:.4f}" for b in blocks.values() for k, v in b.items()
             if isinstance(v, float) and any(f"_{name}" in k for name in BASELINES)]
    return parts + (["Baselines: " + " ".join(named)] if named else [])


def aggregate_saved_runs(pattern: str = "cp_vpe_*_s*.json",
                         out_name: str = "cp_vpe_gpu_aggregate.json", force: bool = False):
    out_path = C.result_path(out_name, force)   # a derived aggregate is regenerated only with the flag
    paths = sorted(glob.glob(os.path.join(C.RES, pattern)))
    by_ds = collections.defaultdict(list)
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        by_ds[d["ds"]].append(d)

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return dict(mean=None, sd=None, n=0)
        return dict(mean=float(np.mean(xs)),
                    sd=(float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0),
                    n=len(xs))

    agg = {}
    for ds, recs in sorted(by_ds.items()):
        agg[ds] = dict(
            seeds=[r["seed"] for r in recs],
            temporal_encoder="SCEM",
            direct_laundering=dict(
                gain_plain_edge=ms([r["direct_laundering"]["gain_plain_edge"] for r in recs]),
                gain_vpe_edge=ms([r["direct_laundering"]["gain_vpe_edge"] for r in recs]),
                gain_cp_vpe_edge=ms([r["direct_laundering"]["gain_cp_vpe_edge"] for r in recs]),
                reduction_plain_vpe=ms([r["direct_laundering"]["reduction_plain_vpe"] for r in recs]),
                reduction_cp_vpe=ms([r["direct_laundering"]["reduction_cp_vpe"] for r in recs]),
                d_t1_mcc_cp_vs_plain=ms([r["direct_laundering"]["d_t1_mcc_cp_vs_plain"] for r in recs]),
                linkage_auc_plain_vpe=ms([r["direct_laundering"]["linkage_plain_vpe"]["auc"]["combined"] for r in recs]),
                linkage_auc_cp_vpe=ms([r["direct_laundering"]["linkage_cp_vpe"]["auc"]["combined"] for r in recs]),
            ),
            inject_then_erase_whitewash=dict(
                gain_plain_vpe=ms([r["inject_then_erase_whitewash"]["gain_plain_vpe"] for r in recs]),
                gain_cp_vpe=ms([r["inject_then_erase_whitewash"]["gain_cp_vpe"] for r in recs]),
                suppression_ratio=ms([r["inject_then_erase_whitewash"]["suppression_ratio"] for r in recs]),
            ),
            inject_then_erase_badmouth=dict(
                drop_plain_vpe=ms([r["inject_then_erase_badmouth"]["drop_plain_vpe"] for r in recs]),
                drop_cp_vpe=ms([r["inject_then_erase_badmouth"]["drop_cp_vpe"] for r in recs]),
                suppression_ratio=ms([r["inject_then_erase_badmouth"]["suppression_ratio"] for r in recs]),
            ),
        )
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print("Wrote", out_path)
    return agg


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CPPR (CP-VPE) evaluation of one seed, or --aggregate the stored seeds.")
    ap.add_argument("--ds", default="bitcoinalpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--rho", type=float, default=0.02)
    ap.add_argument("--granularity", type=int, default=1)
    ap.add_argument("--k-fake", type=int, default=3, help="fakes per target group (the Sybil region's size)")
    ap.add_argument("--n-edge", type=int, default=15)
    ap.add_argument("--n-acc", type=int, default=5)
    ap.add_argument("--out", default=None, help="result file under results/laundering/ or an absolute path; "
                    "required when the protocol differs from the stored runs'")
    ap.add_argument("--gate-norm", choices=GATE_NORMS, default="count",
                    help="credibility-gate normalization: 'count' (CPPR's gate, Eq. 15; the default, ruling G1) "
                         "or 'snapshot' (the ablation of Table S7, used by the stored cp_vpe_* runs)")
    ap.add_argument("--aggregate", action="store_true",
                    help="Aggregate results/laundering/cp_vpe_*_s*.json, or the "
                         "cpgate_<norm>_*_s*.json files when --gate-norm is not 'snapshot'")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result, dump or aggregate file (or set KEEPER_OVERWRITE=1)")
    p = ap.add_argument_group("protocol (Task 3.2, keeper.cppr.protocol; the defaults are the stored runs')")
    p.add_argument("--tiers", choices=TIER_METHODS, default="bfs", help="legacy seeded endorsement or BTP tiers")
    p.add_argument("--btp-eps", type=float, default=None, help="BTP's eps; needed by btp tiers")
    p.add_argument("--btp-edge-cap", type=parse_edge_cap, default=CT.KEEPER_EDGE_CAP_UNITS, metavar="{none,UNITS}",
                   help="per-endorsement cap in theta0 units (default: KEEPER's); 'none' is the uncapped ablation")
    p.add_argument("--btp-neg-from", choices=CT.NEG_FROM, default=CT.KEEPER_NEG_FROM,
                   help="negative ratings BTP's guard counts: trusted raters' (default, ruling R56) or all raters'")
    p.add_argument("--attack-edges", type=int, default=0, metavar="G", help="honest ratings into each group's fakes")
    p.add_argument("--endorser-select", choices=ENDORSER_SELECTIONS, default="random",
                   help="random Tier >= 2 endorsers, or the max-B(u) accounts")
    p.add_argument("--sybil-topology", choices=SYBIL_TOPOLOGIES, default="ring", help="the whitewash region's shape")
    p.add_argument("--pool-eps", type=float, default=None, metavar="EPS",
                   help="eps of the endorser protocol's reference capped calibration (rulings R43, R44); needed with "
                        "attack edges, and equal to --btp-eps in a capped BTP run")
    p.add_argument("--base-cache", metavar="DIR", default=None,
                   help="store the trained base model under DIR, or load it from there (outputs unchanged); by "
                        "convention <repo>/.keeper-cache/base, never under results/")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS), help=f"comma list from {', '.join(ARMS)}")
    p.add_argument("--methods", default=",".join(METHODS),
                   help=f"comma list from {', '.join(METHODS + BASELINES)} to retrain (cppr or a baseline among them); "
                        "every release of the stored methods is still built")
    p.add_argument("--control-arm", action="store_true", help="add the control arm (whitewash without fakes)")
    p.add_argument("--hold-test-snapshot", action="store_true", help="score every model on the original test snapshot")
    p.add_argument("--ungated-baseline", action="store_true", help="also train and score the ungated backbone")
    p.add_argument("--dump-targets", metavar="DIR", default=None, help="write DIR/<result>_targets.json")
    p.add_argument("--dump-verdicts", metavar="DIR", default=None, help="write DIR/<result>_verdicts.json")
    p.add_argument("--prereg", metavar="PATH", default=None, help="record this preregistration file's sha256")
    p.add_argument("--backbone", choices=BACKBONES, default=BACKBONES[0], help="seq_scem (default, the stored runs') "
                   "or rating_scem, the rating-level LocalGDTE-SCEM with M-B (Task 4.7, keeper.cppr.rating_backbone)")
    return ap


def protocol_from_args(a) -> RunProtocol:
    """The RunProtocol of parsed command-line arguments (ValueError for an invalid combination)."""
    return RunProtocol(tiers=a.tiers, btp_eps=a.btp_eps, btp_edge_cap=a.btp_edge_cap, btp_neg_from=a.btp_neg_from,
                       attack_edges=a.attack_edges, endorser_select=a.endorser_select, sybil_topology=a.sybil_topology,
                       arms=parse_arms(a.arms, control=a.control_arm), methods=parse_methods(a.methods),
                       hold_test_snapshot=a.hold_test_snapshot,
                       ungated_baseline=a.ungated_baseline, dump_targets=a.dump_targets,
                       dump_verdicts=a.dump_verdicts, prereg=a.prereg, pool_eps=a.pool_eps, base_cache=a.base_cache,
                       backbone=a.backbone)


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.aggregate:
        if a.gate_norm == "snapshot":
            aggregate_saved_runs(force=a.force)
        else:
            aggregate_saved_runs(f"cpgate_{a.gate_norm}_*_s*.json",
                                 f"cpgate_{a.gate_norm}_aggregate.json", force=a.force)
    else:
        try:
            protocol = protocol_from_args(a)
        except ValueError as err:
            ap.error(str(err))
        run_single_seed(a.ds, a.seed, epochs=a.epochs, rho=a.rho,
                        g=a.granularity, k_fake=a.k_fake,
                        n_edge=a.n_edge, n_acc=a.n_acc, out_name=a.out,
                        gate_norm=a.gate_norm, force=a.force, protocol=protocol)


if __name__ == "__main__":
    main()
