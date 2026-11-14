"""The rating-level backbone of the CPPR driver keeper.cppr.cp_vpe_gpu (--backbone rating_scem; Task 4.7, experiment E7;
protocol docs/preregistration.md Section 2.9).

--backbone seq_scem, the default, is the driver's backbone as before: train_gated_model builds core's LocalGDTE and
the protected attach_scem_and_credibility_gate replaces its S3R by a SCEM that reads the snapshot sequence only.
Backbone then gives no trainer and no config field, so the driver calls train_gated_model as it did
(tests/test_cppr_baselines.py, B6).

--backbone rating_scem trains every model of a run (base, Plain Erasure, Plain VPE, CPPR) as train_gated_model does,
with one change: the model is the archived rating-level LocalGDTE-SCEM (keeper.backbone.local_model_scem), built
through core.make_model with the class set as in keeper.scem.stage_b.cmd_model and M-B on
(keeper.scem.e2_driver.build_model, model "scem": phi learned, N = 2, hazard on; the built module's settings are
checked). The protected attach function then installs CPPR's gate on the model's structural conv layers. Its
isinstance(model.s3r, SCEM) branch keeps the rating-level module, since the configured class subclasses
keeper.scem.scem.SCEM; check_gated asserts both.

M-B witness levels. Per arm, keeper.scem.witness_cred.witness_levels of the arm's training graph before any release
(the graph whose tiers feed CPPR's gate: the base graph for the base model and the direct arm, the injected graph for
an injected arm): causal capped BTP under the per-snapshot calibration frozen on the run's honest training graph
(ruling R32), with the run's eps and KEEPER's cap and guard (keeper.scem.e2_streams.stream_levels). A release appends
nodes, and each takes at every snapshot the tier its gate gives it (extend_levels): a CPPR pool node its pool's tier
(apply_cp_vpe's extended tiers). A Plain VPE window node has tier 0 in its tiers, but the gate weights it by its
pool's mean tier weight, and so does M-B after t_seed (mb_weights). Otherwise the weights are
witness_cred.level_weights: 1 up to t_seed.

Recorded in the result's config block under backbone: the model and its settings, each arm's M-B levels (eps, cap,
guard, the frozen calibration, the levels' digest and per-snapshot tier counts: keeper.scem.e2_driver.mb_record) and,
per trained model in training order, its arm, node counts, the added nodes' tiers, the weights' digest and the learned
phi.
"""
from functools import partial

import numpy as np

from keeper.credibility.tiers import T_SEED, TIER_WEIGHTS
from keeper.cppr.protocol import BACKBONES
from keeper.laundering import core as C
from keeper.scem import e2_driver as E2
from keeper.scem import e2_streams as E
from keeper.scem.witness_cred import level_weights

SEQ, RATING = BACKBONES
SCEM_MODEL = "scem"                 # keeper.scem.e2_driver.SCEM_MODELS: the class of stage_b.cmd_model, phi learned
GATED_NAME = "gated_incoming_edges"  # the gate function attach_scem_and_credibility_gate installs on each conv layer


def extend_levels(levels, cred, n_arm: int) -> np.ndarray:
    """[T, len(cred)] int64: the arm's [T, n_arm] levels, then each node a release appended (ids n_arm, ...) at its gate
    tier cred[node] in every row."""
    levels, cred = np.asarray(levels, dtype=np.int64), np.asarray(cred, dtype=np.int64)
    if levels.ndim != 2 or levels.shape[1] != n_arm or len(cred) < n_arm:
        raise ValueError(f"levels {list(levels.shape)} and {len(cred)} tiers do not fit an arm of {n_arm} accounts")
    added = np.broadcast_to(cred[n_arm:], (levels.shape[0], len(cred) - n_arm))
    return np.concatenate([levels, added], axis=1)


def mb_weights(levels, anon_nodes=None, pool_weights=None) -> np.ndarray:
    """The M-B weights of a model's levels (level_weights); a Plain VPE window node anon_nodes[w] whose pool has the
    weight pool_weights[w] takes it after t_seed, as the gate gives it (attach_scem_and_credibility_gate's override)."""
    weights = level_weights(levels)
    if anon_nodes is not None and pool_weights is not None:
        for w_idx, a_id in enumerate(anon_nodes):
            if int(w_idx) in pool_weights and int(a_id) < weights.shape[1]:
                weights[T_SEED + 1:, int(a_id)] = float(pool_weights[int(w_idx)])
    return weights


def check_gated(model, scem) -> None:
    """RuntimeError unless the attach function kept the rating-level module ``scem`` (as model.scem and model.s3r) and
    installed the gate on every structural conv layer."""
    if model.scem is not scem or model.s3r is not scem:
        raise RuntimeError("attach_scem_and_credibility_gate replaced the rating-level SCEM")
    convs = list(model.structural_layer.conv_layers)
    if not convs or any(getattr(conv.incoming_edges, "__name__", None) != GATED_NAME for conv in convs):
        raise RuntimeError("the credibility gate is not installed on every structural conv layer")


def train_rating_model(D, Tr, seed: int, tt: int, cred_levels, weights, epochs: int = 150, plain_vpe_anon_nodes=None,
                       plain_vpe_pool_weights=None, gate_norm: str = "snapshot") -> tuple:
    """(model, its SCEM settings): train_gated_model with the rating-level backbone. Built through core.make_model with
    ``seed`` and the M-B ``weights`` ([tt, Tr.n], or None for the stored witness rule), gated by the protected attach
    function with the given tiers and Plain VPE pool weights, checked (check_gated), then trained by core.train."""
    from keeper.cppr import cp_vpe_gpu as CV          # the driver imports this module: a top-level import is circular
    if tt != D.tt:
        raise ValueError(f"tt = {tt}, but core.make_model builds the model on D.tt = {D.tt} snapshots")
    m, settings = E2.build_model(D, Tr, seed, SCEM_MODEL, weights)
    scem = m.scem
    CV.attach_scem_and_credibility_gate(m, cred_levels, CV.CRED_WEIGHTS, plain_vpe_anon_nodes=plain_vpe_anon_nodes,
                                        plain_vpe_pool_weights=plain_vpe_pool_weights, gate_norm=gate_norm)
    check_gated(m, scem)
    return C.train(m, Tr, tt, epochs=epochs), settings


def learned_phi(model):
    """The model's learned SCEM clock rate phi, or None for a model without one."""
    phi = getattr(getattr(model, "scem", None), "phi", None)
    return None if phi is None else float(phi)


class Backbone:
    """The run's backbone (RunProtocol.backbone) over its honest training graph Tr. trainer(G_arm, arm) is the trainer
    of an arm's models, with train_gated_model's signature, or None under seq_scem; config() is the result's config
    field ({} under seq_scem)."""

    def __init__(self, protocol, Tr, tt: int):
        self.name, self.eps, self.Tr, self.tt = protocol.backbone, protocol.btp_eps, Tr, int(tt)
        self.arms, self.models, self.settings = {}, [], None

    def trainer(self, G_arm, arm: str):
        """None under seq_scem. Under rating_scem, the trainer of the models of an arm whose graph before any release is
        G_arm: its M-B levels are computed here and recorded under ``arm``."""
        if self.name == SEQ:
            return None
        Tr_arm = C.train_split(G_arm, self.tt)
        levels = E.stream_levels(Tr_arm, self.Tr, self.eps, self.tt)
        self.arms[arm] = E2.mb_record(levels, levels, self.Tr, self.eps, None, None)
        return partial(self._train, arm, levels)

    def _train(self, arm: str, levels, D, Tr, seed: int, tt: int, cred_levels, epochs: int = 150,
               plain_vpe_anon_nodes=None, plain_vpe_pool_weights=None, gate_norm: str = "snapshot"):
        full = extend_levels(levels, cred_levels, levels.shape[1])
        weights = mb_weights(full, plain_vpe_anon_nodes, plain_vpe_pool_weights)
        m, self.settings = train_rating_model(D, Tr, seed, tt, cred_levels, weights, epochs, plain_vpe_anon_nodes,
                                              plain_vpe_pool_weights, gate_norm)
        added = full[-1, levels.shape[1]:]
        self.models.append(dict(arm=arm, n_nodes=int(full.shape[1]), n_added=int(len(added)),
                                added_tier_counts=np.bincount(added, minlength=len(TIER_WEIGHTS)).tolist(),
                                weights_sha256=E2.sha256_array(weights), learned_phi=learned_phi(m)))
        return m

    def config(self) -> dict:
        if self.name == SEQ:
            return {}
        return dict(backbone=dict(
            name=self.name, model=SCEM_MODEL, module="keeper.backbone.local_model_scem.LocalGDTE",
            built="core.make_model, class stage_b.scem_localgdte(stage_b.configured_scem(witness_weights=M-B))",
            settings=self.settings,
            mb=dict(levels="causal capped BTP of the arm's training graph before any release, calibration frozen on "
                           "the honest training graph; release nodes at their gate tier (Plain VPE: its pool weight)",
                    arms=self.arms),
            models=self.models))
