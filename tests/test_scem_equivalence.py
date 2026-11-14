"""The one SCEM engine, keeper.scem.scem, against the user's original module.

keeper/scem/scem.py is the user's scem.py with approved changes. On 2026-10-17, four in-place
index_add_ calls in SCEM._forward_graph_stream became out-of-place, so that the module can be
trained. Task 3.3 (approved 2026-10-26: upgrade M-B and the E1 release policies) added options whose
defaults keep the old behaviour; tests/test_scem_upgrades.py checks those defaults against the stored
results. Task 3.4 (approved 2026-10-26: upgrade G5) added the module option phi_fixed, whose default None
keeps the module unchanged; tests/test_scem_phi_fixed.py checks that default against the engine with the
G5 lines reverted. Task 4.5 (the E2 no-hazard ablation, approved in the protocol) added the module option hazard,
whose default True keeps the module unchanged; tests/test_e2_driver.py checks that default against the engine with the
E2 lines reverted. The original is thecode/scem.py at tag pre-revision-2026-10-26, read with `git show`.

1. Source: the engine differs from the original only in the approved edits listed in APPROVED_EDITS
   (and in main(), whose path defaults Task 1.3 changed).
2. Training: one epoch of the user's training loop back-propagates through the engine; with the
   in-place lines the backward pass fails ("modified by an inplace operation").
3. Forward: inside the user's archived LocalGDTE and from the same seed, the engine and the original
   give identical initial parameters, embeddings and auxiliary loss. This runs on the device of
   keeper.laundering.core: the CPU here (with the project's override FEDUNL_ALLOW_CPU=1), the GPU on a
   GPU host.
4. The `equiv` CLI needs `--original FILE`, compares that file's SCEM class with the engine, and
   replaces an existing record only with `--force`. Without an original both arms would be the
   engine, and the record would look like evidence without being any.
5. Stored value (GPU only, Task 1.6): forward_check reproduces the auxiliary loss of
   results/scem/stage_b/equiv_bitcoinalpha_s0.json, which an NVIDIA RTX A6000 computed in
   deterministic mode.
"""
import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from keeper.scem import stage_b  # first: it sets CUBLAS_WORKSPACE_CONFIG before CUDA starts
from keeper.laundering import core as C
from keeper.scem import scem as SC

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "keeper" / "scem" / "scem.py"
ORIGINAL = "pre-revision-2026-10-26:thecode/scem.py"
STORED = ROOT / "results" / "scem" / "stage_b" / "equiv_bitcoinalpha_s0.json"
EQUIV_RECORD = STORED.name
# The source pin below enforces the user's rule in code: the engine may differ from the original only
# by approved edits. A later approved change (Task 3.3, for example) adds its edit to APPROVED_EDITS, or
# extends the comparison; it must not delete the pin.
# Task 3.4 (approved 2026-10-26, upgrade G5): the SCEM module option phi_fixed. tests/test_scem_phi_fixed.py reverts
# these lines alone, to compare the module's forward pass at phi_fixed=None with the engine before them.
G5_EDITS = [  # (original text, engine text)
    ("            closed_only_reads: bool = False,\n"
     "            **kwargs,",
     "            closed_only_reads: bool = False,\n"
     "            phi_fixed: Optional[float] = None,\n"
     "            **kwargs,"),
    ("            self.phi_min = 0.25 * self.phi_max\n"
     "            self.horizon_A = erlang_horizon_A(self.N, self.phi_min, q=self.q_quantile)\n"
     "\n"
     "            self.phi_raw = nn.Parameter(torch.tensor(0.0))",
     "            self.phi_min = 0.25 * self.phi_max\n"
     "            # Upgrade G5 (Task 3.4): phi_fixed fixes phi inside the learned range [phi_min, phi_max], a\n"
     "            # constant without gradient, and the forced-close age A = erlang_horizon_A(N, phi_fixed);\n"
     "            # None keeps phi learned and A at phi_min.\n"
     "            if phi_fixed is not None and not self.phi_min <= phi_fixed <= self.phi_max:\n"
     "                raise ValueError(f\"phi_fixed must lie in the learned range [phi_min, phi_max] = \"\n"
     "                                 f\"[{self.phi_min}, {self.phi_max}], got {phi_fixed!r}\")\n"
     "            self.phi_fixed = None if phi_fixed is None else float(phi_fixed)\n"
     "            self.horizon_A = erlang_horizon_A(\n"
     "                self.N, self.phi_min if self.phi_fixed is None else self.phi_fixed, q=self.q_quantile)\n"
     "\n"
     "            self.phi_raw = nn.Parameter(torch.tensor(0.0), requires_grad=self.phi_fixed is None)"),
    ("        def phi(self) -> torch.Tensor:\n"
     "            return self.phi_min + (self.phi_max - self.phi_min) * torch.sigmoid(self.phi_raw)",
     "        def phi(self) -> torch.Tensor:\n"
     "            if self.phi_fixed is not None:  # upgrade G5: the fixed rate, without gradient\n"
     "                return torch.tensor(self.phi_fixed, dtype=self.phi_raw.dtype, device=self.phi_raw.device)\n"
     "            return self.phi_min + (self.phi_max - self.phi_min) * torch.sigmoid(self.phi_raw)"),
]
# Task 4.5 (E2, approved ablation of the protocol, Section 2.4): the SCEM module option hazard. Its default True keeps
# the module unchanged; hazard=False gives lambda = 0. tests/test_e2_driver.py reverts these lines alone, to compare
# the module's forward pass and gradients at the default with the engine before them.
E2_EDITS = [  # (original text, engine text)
    ("            L_h: float = 1.0,\n"
     "            closed_only_reads: bool = False,",
     "            L_h: float = 1.0,\n"
     "            hazard: bool = True,\n"
     "            closed_only_reads: bool = False,"),
    ("            self.closed_only_reads = bool(closed_only_reads)\n"
     "\n"
     "            self.phi_max",
     "            self.closed_only_reads = bool(closed_only_reads)\n"
     "            self.hazard = bool(hazard)  # E2 ablation (Task 4.5): False sets the hazard lambda to 0 (lambda_max = 0)\n"
     "\n"
     "            self.phi_max"),
    ("            nu = F.softplus(self.h_theta(xi, b_emit))\n"
     "            gam = self.gamma_sigma.unsqueeze(0)",
     "            nu = F.softplus(self.h_theta(xi, b_emit))\n"
     "            if not self.hazard:  # E2 ablation: lambda = 0, the forward pass of lambda_max = 0 without its 0/0 "
     "gradient\n"
     "                return nu, torch.zeros_like(nu)\n"
     "            gam = self.gamma_sigma.unsqueeze(0)"),
]
APPROVED_EDITS = [  # (original text, engine text)
    # 2026-10-17, SCEM._forward_graph_stream: out-of-place index_add, so that the module can be trained.
    ("n_dur.index_add_(0, v_r, n_inc)", "n_dur = n_dur.index_add(0, v_r, n_inc)"),
    ("m_dur.index_add_(0, v_r, m_inc)", "m_dur = m_dur.index_add(0, v_r, m_inc)"),
    ("n_closed.index_add_(0, v_r, n_tot)", "n_closed = n_closed.index_add(0, v_r, n_tot)"),
    ("m_closed.index_add_(0, v_r, m_tot_rec)", "m_closed = m_closed.index_add(0, v_r, m_tot_rec)"),
    # Task 3.3 (approved 2026-10-26). Imports of the new modules.
    ("from scipy.special import gammainccinv\n\ntry:",
     "from scipy.special import gammainccinv\n\n"
     "from keeper.credibility.tiers import T_SEED, TIER_WEIGHTS\n"
     "from keeper.scem.policies import record_masses, release_policy\n"
     "from keeper.scem.witness_cred import check_witness_levels, check_witness_weights, mb_witness_kappa\n\n"
     "try:"),
    # run_scem_streaming_weights_np: the new options and their documentation.
    ("    closed_only_reads: bool = False,\n) -> Dict[str, np.ndarray]:",
     "    closed_only_reads: bool = False,\n"
     "    witness_cred: Optional[np.ndarray] = None,\n"
     "    t_seed: int = T_SEED,\n"
     "    tier_weights: Tuple[float, ...] = TIER_WEIGHTS,\n"
     "    policy: str = \"erlang\",\n"
     "    quarantine_D: float = 4,\n"
     "    return_records: bool = False,\n"
     ") -> Dict[str, np.ndarray]:"),
    # The release policy replaces the hard-wired Erlang horizon; witness levels and trajectories are set up.
    ("        - monotone_verified: bool confirming omega_k(t) never decreased for any edge\n"
     "    \"\"\"\n"
     "    E = len(edges)\n"
     "    A = erlang_horizon_A(N, phi, q=q_quantile)\n",
     "        - monotone_verified: bool confirming omega_k(t) never decreased for any edge\n"
     "        - records (with return_records): per record the psi-mixed omega, pi, revoked_haz, revoked_sup,\n"
     "          revoked (their sum) and closed, omega_by_snap / pi_by_snap [K_snap, E], the released and the\n"
     "          provisional mass at the end of each snapshot, and witness_kappa, the witness reliability the\n"
     "          rating registered its evidence with\n"
     "\n"
     "    Task 3.3 options leave the engine unchanged, bit for bit, at their defaults: witness_cred, t_seed and\n"
     "    tier_weights weight witnesses by their causal BTP tier (upgrade M-B, keeper/scem/witness_cred.py);\n"
     "    policy and quarantine_D choose the release policy (keeper/scem/policies.py).\n"
     "    \"\"\"\n"
     "    E = len(edges)\n"
     "    rel = release_policy(policy, N, phi, q_quantile, quarantine_D)\n"
     "    A = rel.A\n"
     "    if witness_cred is not None:\n"
     "        witness_cred = check_witness_levels(witness_cred, K_snap, n_nodes, len(tier_weights))\n"
     "    snaps_omega: List[np.ndarray] = []\n"
     "    snaps_pi: List[np.ndarray] = []\n"
     "    witness_kap = np.zeros(E, dtype=np.float64)\n"),
    # O2 supersession: open window and escrowed share from the policy.
    ("                        if not closed_flag[prev_ei] and age_sup < A:\n"
     "                            Q_sup, _ = erlang_Q_P_np(N, phi * age_sup)",
     "                        if not closed_flag[prev_ei] and rel.is_open(age_sup):\n"
     "                            Q_sup = rel.escrowed(age_sup)"),
    # Upgrade M-B: the witness reliability of a rating from the sender's causal tier.
    ("                kap_u = max(float(kappa_now[u_k]), 0.35 if t <= 4 else 0.20)\n"
     "                if s_k < 0:",
     "                kap_u = max(float(kappa_now[u_k]), 0.35 if t <= 4 else 0.20)\n"
     "                if witness_cred is not None:  # upgrade M-B\n"
     "                    kap_u = mb_witness_kappa(float(kappa_now[u_k]), t, int(witness_cred[t, u_k]), t_seed,\n"
     "                                             tier_weights)\n"
     "                witness_kap[e_i] = kap_u\n"
     "                if s_k < 0:"),
    # Segment integration, closure and provisional mass from the policy.
    ("                Qa_v, _ = erlang_Q_P_np(N, phi * xa_v)\n"
     "                Qb_v, _ = erlang_Q_P_np(N, phi * xb_v)",
     "                Qa_v = rel.escrowed(xa_v)\n"
     "                Qb_v = rel.escrowed(xb_v)"),
    ("inc = segment_release_increment_np(N, phi, lam_seg, xa_v, xb_v, La_seg)",
     "inc = rel.released(lam_seg, xa_v, xb_v, La_seg)"),
    ("Q_A, _ = erlang_Q_P_np(N, phi * A)", "Q_A = rel.escrowed(A)"),
    ("Q_now, _ = erlang_Q_P_np(N, phi * age_now)", "Q_now = rel.escrowed(age_now)"),
    # Ruling R57 (fix round 1): under no_commit the tallies behind kappa and V2 read the surviving weight.
    ("                    monotone_ok = False\n"
     "\n"
     "                pos_v = s_v > 0",
     "                    monotone_ok = False\n"
     "\n"
     "                if rel.reads_provisional:  # R57: no_commit's tallies hold each record's surviving weight\n"
     "                    d_omega = d_omega + rel.provisional_read(psi_k[v_idx], pi_ch[v_idx], L_exp_ch[v_idx])\n"
     "                pos_v = s_v > 0"),
    # Per-record output (return_records).
    ("            active_edges = [ei for ei in active_edges if not closed_flag[ei]]\n"
     "\n"
     "        Lambda_hist[t + 1] = Lambda_hist[t] + lam_curr",
     "            active_edges = [ei for ei in active_edges if not closed_flag[ei]]\n"
     "\n"
     "        if return_records:\n"
     "            snaps_omega.append(np.sum(psi_k * omega_ch, axis=-1))\n"
     "            snaps_pi.append(np.sum(psi_k * pi_ch, axis=-1))\n"
     "        Lambda_hist[t + 1] = Lambda_hist[t] + lam_curr"),
    ("    open_rec_mib = float(max_open_records * (d_dim + 2 * d_psi + 6) * 4) / (1024.0 ** 2)\n"
     "    return {",
     "    open_rec_mib = float(max_open_records * (d_dim + 2 * d_psi + 6) * 4) / (1024.0 ** 2)\n"
     "    extra = {}\n"
     "    if return_records:\n"
     "        extra[\"records\"] = record_masses(psi_k, omega_ch, pi_ch, revoked_haz_ch, revoked_sup_ch, closed_flag,\n"
     "                                         snaps_omega, snaps_pi, witness_kap)\n"
     "    return {"),
    ("        \"total_scem_state_mib\": round(node_state_mib + open_rec_mib, 3),\n"
     "    }",
     "        \"total_scem_state_mib\": round(node_state_mib + open_rec_mib, 3),\n"
     "        **extra,\n"
     "    }"),
    # SCEM module: optional [T, n] witness_weights (upgrade M-B) multiply the witness term in both modes.
    ("            node_labels: Optional[torch.Tensor] = None,\n"
     "        ) -> torch.Tensor:\n"
     "            if train_edges is None or train_labels is None or index_list is None:\n"
     "                return self._forward_sequence_mode(x)\n"
     "            return self._forward_graph_stream(x, train_edges, train_labels, index_list, node_labels=node_labels)",
     "            node_labels: Optional[torch.Tensor] = None,\n"
     "            witness_weights: Optional[torch.Tensor] = None,\n"
     "        ) -> torch.Tensor:\n"
     "            if witness_weights is not None:  # upgrade M-B: [T, n] weights of the witness term\n"
     "                check_witness_weights(witness_weights, x.shape[1], x.shape[0])\n"
     "            if train_edges is None or train_labels is None or index_list is None:\n"
     "                return self._forward_sequence_mode(x, witness_weights=witness_weights)\n"
     "            return self._forward_graph_stream(x, train_edges, train_labels, index_list, "
     "node_labels=node_labels,\n"
     "                                              witness_weights=witness_weights)"),
    ("        def _forward_sequence_mode(self, x: torch.Tensor) -> torch.Tensor:",
     "        def _forward_sequence_mode(self, x: torch.Tensor,\n"
     "                                   witness_weights: Optional[torch.Tensor] = None) -> torch.Tensor:"),
    ("                ev_vec = torch.cat([kappa.unsqueeze(-1) * g_vec, s_tilde, v3_zero], dim=-1)",
     "                v1_term = kappa.unsqueeze(-1) * g_vec\n"
     "                if witness_weights is not None:  # upgrade M-B\n"
     "                    v1_term = v1_term * witness_weights[t].to(dev, dtype).unsqueeze(-1)\n"
     "                ev_vec = torch.cat([v1_term, s_tilde, v3_zero], dim=-1)"),
    ("            node_labels: Optional[torch.Tensor] = None,\n"
     "        ) -> torch.Tensor:\n"
     "            B, K, d = x.shape",
     "            node_labels: Optional[torch.Tensor] = None,\n"
     "            witness_weights: Optional[torch.Tensor] = None,\n"
     "        ) -> torch.Tensor:\n"
     "            B, K, d = x.shape"),
    ("                    raw_w_v1 = kappa[u_k].unsqueeze(-1) * g_k\n"
     "                    pending_v1[t + self.L_f].append({",
     "                    raw_w_v1 = kappa[u_k].unsqueeze(-1) * g_k\n"
     "                    if witness_weights is not None:  # upgrade M-B\n"
     "                        raw_w_v1 = raw_w_v1 * witness_weights[t].to(dev, dtype)[u_k].unsqueeze(-1)\n"
     "                    pending_v1[t + self.L_f].append({"),
    *G5_EDITS,  # Task 3.4, upgrade G5
    *E2_EDITS,  # Task 4.5, the E2 no-hazard ablation
]


def original_source() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", ORIGINAL],
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):   # no git, a shallow clone or a missing tag
        pytest.fail("tag pre-revision-2026-10-26 not found; fetch tags")


def load_original(tmp_path):
    """The original module, written to tmp_path and loaded with stage_b's loader (as `equiv --original`)."""
    path = tmp_path / "scem_original.py"
    path.write_text(original_source())
    return stage_b.load_original_module(str(path))


def run_stage_b(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["stage_b.py", *args])
    stage_b.main()


@pytest.fixture
def stage_b_out(monkeypatch, tmp_path):
    """Stage B writes its records to tmp_path, never into results/scem/stage_b/; KEEPER_OVERWRITE unset."""
    monkeypatch.setattr(stage_b, "OUT", str(tmp_path))
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    return tmp_path


@pytest.fixture
def fake_original(tmp_path):
    path = tmp_path / "fake_original.py"
    path.write_text("class SCEM:\n    SOURCE = 'fake_original.py'\n")
    return path


def without_main(source: str) -> str:
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == "main")]
    return ast.dump(tree)


@pytest.fixture
def pipeline_device(monkeypatch):
    """The device keeper.laundering.core chose at import. Without a GPU the project's override lets
    the model tests run on the CPU."""
    if C.DEV.type == "cpu":
        monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")
    return C.DEV.type


def test_engine_differs_from_original_only_in_the_approved_lines():
    engine = ENGINE.read_text()
    for old, new in APPROVED_EDITS:
        assert engine.count(new) == 1 and old not in engine, f"approved edit not merged: {new}"
        engine = engine.replace(new, old)
    assert without_main(engine) == without_main(original_source())


@pytest.mark.filterwarnings(r"ignore:`torch\.(cpu|cuda)\.amp\.autocast:FutureWarning")  # inside torch.utils.checkpoint
def test_engine_back_propagates(pipeline_device, monkeypatch):
    archive = importlib.import_module("keeper.backbone.local_model_scem")
    assert archive.SCEM is SC.SCEM
    D, G, Tr, _ = stage_b.load_stream("bitcoinalpha")
    monkeypatch.setattr(C, "LocalGDTE", archive.LocalGDTE)
    m = C.train(C.make_model(D, Tr, 0, D.tt), Tr, D.tt, epochs=1)
    assert all(bool(torch.isfinite(p).all()) for p in m.parameters())


def test_engine_forward_equals_original(pipeline_device, tmp_path):
    original = load_original(tmp_path)
    assert original.SCEM is not SC.SCEM
    out = stage_b.forward_check("bitcoinalpha", 0, pipeline_device, original_scem=original.SCEM)
    assert set(out) == set(json.loads(STORED.read_text()))
    assert out["identical_initial_parameters"]
    assert out["embeddings_equal"] and out["max_abs_diff_embedding"] == 0.0
    assert out["aux_loss_original"] == out["aux_loss_patched"]


def test_parameter_check_needs_the_same_non_empty_keys():
    w = torch.zeros(2)
    assert stage_b.same_parameters({"w": w}, {"w": w.clone()})
    assert not stage_b.same_parameters({}, {})
    assert not stage_b.same_parameters({"w": w}, {"w": w, "b": w})
    assert not stage_b.same_parameters({"w": w}, {"w": w + 1})


def test_equiv_cli_requires_an_original(stage_b_out, monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        run_stage_b(monkeypatch, "equiv")
    err = capsys.readouterr().err
    assert stop.value.code == 2
    assert "git show pre-revision-2026-10-26:thecode/scem.py > FILE" in err
    assert "tests/test_scem_equivalence.py" in err
    assert not any(stage_b_out.iterdir())


def test_equiv_cli_compares_with_the_given_original(stage_b_out, fake_original, monkeypatch):
    seen = {}

    def fake_forward_check(ds, seed, device, original_scem=None):
        seen["original_scem"] = original_scem
        return {"ds": ds, "seed": seed}

    monkeypatch.setattr(stage_b, "forward_check", fake_forward_check)
    run_stage_b(monkeypatch, "equiv", "--original", str(fake_original))
    assert seen["original_scem"].SOURCE == "fake_original.py"
    assert json.loads((stage_b_out / EQUIV_RECORD).read_text()) == {"ds": "bitcoinalpha", "seed": 0}


def test_equiv_cli_refuses_to_overwrite_a_record(stage_b_out, fake_original, monkeypatch, capsys):
    record = stage_b_out / EQUIV_RECORD
    record.write_text("stored")
    with pytest.raises(SystemExit) as stop:      # refused before any model work
        run_stage_b(monkeypatch, "equiv", "--original", str(fake_original))
    assert stop.value.code == 2 and "exists; pass --force" in capsys.readouterr().err
    assert record.read_text() == "stored"
    monkeypatch.setattr(stage_b, "forward_check", lambda ds, seed, device, original_scem=None: {"ds": ds})
    run_stage_b(monkeypatch, "equiv", "--original", str(fake_original), "--force")
    assert json.loads(record.read_text()) == {"ds": "bitcoinalpha"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (the NVIDIA RTX A6000 runs of Task 1.6)")
def test_forward_check_reproduces_stored_aux_loss_on_gpu():
    stored = json.loads(STORED.read_text())
    assert stored["aux_loss_patched"] == 0.04037242382764816
    out = stage_b.forward_check("bitcoinalpha", 0, "cuda")      # without an original only aux_loss_patched counts
    assert out["aux_loss_patched"] == stored["aux_loss_patched"]
