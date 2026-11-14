"""Task 3.3: the SCEM engine's and module's upgrades. Witness levels, the witness attack and Property 1's bound
are in tests/test_scem_witness_attack.py; shared helpers in tests/scem_common.py.

1. Regression: with every new option at its default, the engine reproduces the stored Stage B records
   (results/scem/stage_b/engine_*.json) value for value, apart from each configuration's wall-clock seconds
   and the run's provenance. M-B with every account in Tier 3 is the stored rule, bit for bit.
2. M-B (design-M-B.md, Property 1). Five fresh Sybils bad-mouth an honest warner W one snapshot after W warned
   T. Under the stored witness rule they revoke 44.0% of the warning: the review's closed form, which
   keeper.analysis.defense_extra matches to the engine. With M-B and Tier-0 Sybils, the revoked share stays
   within the analytic bound (xi <= 5 w_0 kappa_max through the hazard map) and is numerically 0.
3. Window property (Property 2): evidence maturing at or after t_k + A leaves the record unchanged.
4. no_commit revokes after A where erlang does not, and reads the surviving weight retroactively (ruling R57).
   5. arrival revokes nothing. 6. quarantine commits at D.
7. Mass conservation and monotone released mass per record, for every policy.
8. The SCEM module: witness_weights=None and all-ones weights leave the forward pass unchanged, in both modes
   and inside the archived LocalGDTE (stage_b.forward_check). CPU only.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from keeper.scem import stage_b  # first: it sets CUBLAS_WORKSPACE_CONFIG before CUDA starts
from keeper.analysis import defense_extra as DE
from keeper.laundering import core as C
from keeper.scem import policies as RP
from keeper.scem import scem as SC
from keeper.scem import witness_cred as WC
from tests.scem_common import (HISTORY, N_NODES, PSI_NEG, SYBILS, TOL, W0, W3, T, W, closed_form, engine,
                               sybil_levels, sybil_stream)

ROOT = Path(__file__).resolve().parents[1]
STAGE_B = ROOT / "results" / "scem" / "stage_b"
NETWORKS = ("bitcoinalpha", "bitcoinotc", "wiki-rfa")


# ------------------------------------------------------------------ 1. regression


def drop_run_fields(record):
    """A Stage B engine record without what changes from run to run: each configuration's wall-clock seconds
    and the provenance (host, time, argv, code hash). The provenance's dataset fields stay."""
    out = {key: value for key, value in record.items() if key != "provenance"}
    out["engine"] = {name: {key: value for key, value in cfg.items() if key != "seconds"}
                     for name, cfg in record["engine"].items()}
    out["data"] = {key: record["provenance"][key] for key in ("dataset", "data_md5", "signed_meta")}
    return out


@pytest.mark.parametrize("ds", NETWORKS)
def test_default_engine_reproduces_the_stored_stage_b_records(ds, tmp_path, monkeypatch):
    monkeypatch.setattr(stage_b, "OUT", str(tmp_path))          # never write into results/ (ruling R27)
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    stage_b.cmd_engine(ds)
    new = json.loads((tmp_path / f"engine_{ds}.json").read_text())
    stored = json.loads((STAGE_B / f"engine_{ds}.json").read_text())
    assert drop_run_fields(new) == drop_run_fields(stored)


def test_explicit_defaults_and_all_tier3_mb_give_the_stored_engine_bit_for_bit():
    """Bitcoin-Alpha's training stream. M-B with every account in Tier 3 (weight 1) keeps max(kappa, 0.35) up
    to t_seed = 4 and gives 1.0 * max(kappa, 0.20) after it, which is the stored rule."""
    D, G, Tr, sign = stage_b.load_stream("bitcoinalpha")
    n, K = int(max(G.n, Tr.edges.max() + 1)), int(Tr.snap.max()) + 1
    base = SC.run_scem_streaming_weights_np(Tr.edges, sign, Tr.snap, n, K_snap=K)
    explicit = SC.run_scem_streaming_weights_np(Tr.edges, sign, Tr.snap, n, K_snap=K, witness_cred=None,
                                                policy="erlang", quarantine_D=4, return_records=True)
    tier3 = SC.run_scem_streaming_weights_np(Tr.edges, sign, Tr.snap, n, K_snap=K,
                                             witness_cred=np.full((K, n), 3, dtype=np.int64))
    assert set(explicit) == set(base) | {"records"} and set(tier3) == set(base)
    for out in (explicit, tier3):
        for key, value in base.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(out[key], value), key
            else:
                assert out[key] == value, key
    rec = explicit["records"]
    assert np.array_equal(np.clip(rec["omega"], 0.0, 1.0), base["omega_final"])
    assert np.array_equal(np.clip(rec["pi"], 0.0, 1.0), base["pi_final"])
    assert np.array_equal(np.clip(rec["revoked_haz"], 0.0, 1.0), base["revoked_haz_final"])
    assert np.array_equal(rec["omega_by_snap"][-1], rec["omega"]) and rec["omega_by_snap"].shape == (K, len(sign))


# ------------------------------------------------------------------ M-B witness rule


@pytest.mark.parametrize("kappa, t, level, want", [
    (0.10, 3, 0, 0.35), (0.50, 4, 0, 0.50), (0.10, 4, 3, 0.35),            # t <= t_seed: max(kappa, 0.35)
    (0.10, 5, 2, 0.70 * 0.20), (0.60, 5, 3, 0.60), (0.60, 5, 1, 0.40 * 0.60),  # Tier >= 1: w max(kappa, 0.20)
    (0.60, 5, 0, 0.01 * 0.60), (0.00, 9, 0, 0.0)])                         # Tier 0: w_0 kappa, no floor
def test_mb_witness_reliability_by_hand(kappa, t, level, want):
    assert WC.mb_witness_kappa(kappa, t, level) == pytest.approx(want, abs=1e-15)


def test_mb_bootstrap_window_follows_t_seed():
    assert WC.mb_witness_kappa(0.1, 6, 0, t_seed=6) == 0.35
    assert WC.mb_witness_kappa(0.1, 7, 0, t_seed=6) == pytest.approx(0.001, abs=1e-15)
    assert WC.mb_witness_kappa(0.1, 7, 2, t_seed=6, tier_weights=(0.0, 0.5, 0.5, 0.5)) == pytest.approx(0.1)


def test_witness_levels_are_checked():
    good = np.zeros((21, 5), dtype=np.int64)
    assert np.array_equal(WC.check_witness_levels(good, 21, 5), good)
    assert WC.check_witness_levels(np.zeros((22, 6), dtype=np.int32), 21, 5).dtype == np.int64
    for bad in (np.zeros((20, 5), dtype=np.int64), np.zeros((21, 4), dtype=np.int64), np.zeros(21, dtype=np.int64),
                np.full((21, 5), 4), np.full((21, 5), -1), np.zeros((21, 5))):
        with pytest.raises(ValueError):
            WC.check_witness_levels(bad, 21, 5)
    with pytest.raises(ValueError):                              # the engine checks them before it runs
        engine([(0, 1, -1, 5)], 2, witness_cred=np.zeros((20, 2), dtype=np.int64))


# ------------------------------------------------------------------ 2. adversarial witnesses (Property 1)


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_five_fresh_sybils_revoke_the_closed_form_share_without_mb(lag):
    rows, k = sybil_stream(lag=lag)
    rec = engine(rows, N_NODES)["records"]
    assert rec["pi"][k] == 0.0 and rec["closed"][k]                # the warning has closed by snapshot 21
    assert abs(rec["revoked"][k] - closed_form(lag, [0.20] * 5)) < TOL
    if lag == 1:
        # The review's 44.0% +- 0.5%: five witnesses at the 0.20 floor. The engine gives 43.97% (58.07% at lag 0).
        assert abs(rec["revoked"][k] - 0.440) <= 0.005


@pytest.mark.parametrize("ring", [False, True])
def test_mb_tier0_sybils_stay_within_the_analytic_bound(ring):
    rows, k = sybil_stream(ring=ring)
    got = engine(rows, N_NODES, witness_cred=sybil_levels())["records"]["revoked"][k]
    bound = closed_form(1, [W0 * 1.0] * 5)        # xi <= L_h * 5 * w_0 * kappa_max, kappa_max = 1: 4.03%
    assert got <= bound
    if not ring:
        assert got < TOL                           # fresh Sybils have kappa 0, so their M-B weight w_0 kappa is 0
    else:
        p = float(SC.erlang_Q_P_np(2, 0.5 * 2.0)[1])   # each Sybil's ring rating, released by t_k + 1 (age 2)
        kappa = (p + 1.0) / (p + 2.0) * (1.0 - math.exp(-p / 2.0))
        assert 0.0 < got < 0.01                     # numerically ~0, and the bound is not vacuous
        assert abs(got - closed_form(1, [W0 * kappa] * 5)) < TOL


@pytest.mark.parametrize("n_endorsed", [0, 1, 2, 5])
def test_mb_hazard_is_bounded_by_the_sybils_tiers(n_endorsed):
    """Property 1: L_h c_max (N_T>=1 + w_0 N_T0) bounds the hazard; here c_max = 1, and fresh Sybils in Tier 3
    carry w_3 * 0.20 each, Tier-0 ones nothing."""
    rows, k = sybil_stream()
    levels = sybil_levels()
    levels[:, list(SYBILS[:n_endorsed])] = 3
    got = engine(rows, N_NODES, witness_cred=levels)["records"]["revoked"][k]
    assert abs(got - closed_form(1, [W3 * 0.20] * n_endorsed)) < TOL
    assert got <= closed_form(1, [1.0] * n_endorsed + [W0] * (len(SYBILS) - n_endorsed))


def test_records_carry_the_witness_reliability_each_rating_registered_with():
    rows, k = sybil_stream()
    stored = engine(rows, N_NODES)["records"]["witness_kappa"]
    mb = engine(rows, N_NODES, witness_cred=sybil_levels())["records"]["witness_kappa"]
    bad = np.arange(len(rows) - len(SYBILS), len(rows))
    assert np.all(stored[bad] == 0.20) and np.all(mb[bad] == 0.0)     # fresh Sybils after t_seed: floor / w_0 * 0
    assert np.all(stored[:2] == 0.35) and np.all(mb[:2] == 0.35)      # snapshot 0: the bootstrap floor either way
    assert stored[k] >= 0.20 and mb[k] == pytest.approx(0.70 * stored[k])  # W at Tier 2: w_2 * max(kappa, 0.20)


# ------------------------------------------------------------------ release policies


def test_release_policies_are_validated():
    assert RP.POLICIES == ("erlang", "arrival", "quarantine", "no_commit")
    with pytest.raises(ValueError, match="release policy"):
        RP.release_policy("exponential")
    for D in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError, match="quarantine_D"):
            RP.release_policy("quarantine", quarantine_D=D)
    with pytest.raises(ValueError, match="release policy"):
        engine([(0, 1, -1, 5)], 2, policy="retroactive")


def test_erlang_policy_is_the_engine_closed_form():
    p = RP.release_policy("erlang", N=2, phi=0.5, q=0.99)
    assert p.A == SC.erlang_horizon_A(2, 0.5, q=0.99)
    x, lam, L = np.linspace(0, 14, 29), np.linspace(0, 2, 29), np.linspace(0, 1, 29)
    assert np.array_equal(p.escrowed(x), SC.erlang_Q_P_np(2, 0.5 * x)[0])
    assert np.array_equal(p.released(lam, x, x + 1, L), SC.segment_release_increment_np(2, 0.5, lam, x, x + 1, L))
    assert p.is_open(13.0) and not p.is_open(p.A)


def test_policy_closure_ages_and_supersession_windows():
    arrival, quarantine, no_commit = (RP.release_policy(name) for name in ("arrival", "quarantine", "no_commit"))
    assert arrival.A == 1.0 and not arrival.is_open(0.0)
    assert quarantine.A == 4.0 and quarantine.is_open(3.9) and not quarantine.is_open(4.0)
    assert no_commit.A == math.inf and no_commit.is_open(1e9)
    x = np.array([0.0, 0.5, 3.0, 4.0])
    assert arrival.escrowed(x).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert quarantine.escrowed(x).tolist() == no_commit.escrowed(x).tolist() == [1.0] * 4
    zero = np.zeros(4)
    assert arrival.released(zero + 2.0, x, x + 1, zero).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert not quarantine.released(zero + 2.0, x, x + 1, zero).any()


WINDOW_POLICIES = [("erlang", {}), ("erlang", dict(N=1, phi=0.25)), ("erlang", dict(N=5, phi=1.25)),
                   ("quarantine", {}), ("quarantine", dict(quarantine_D=2.5))]


@pytest.mark.parametrize("policy, kw", WINDOW_POLICIES)
def test_evidence_maturing_at_or_after_the_horizon_leaves_the_record_unchanged(policy, kw):
    """Property 2. Sybils bad-mouth W at every lag; their evidence matures at t_k + lag + L_f."""
    t_k, K, L_f = 5, 40, 1
    A = RP.release_policy(policy, **kw).A
    warning = [(W, T, -1, t_k)]
    clean = engine(warning, N_NODES, K, policy=policy, **kw)["records"]
    inside = outside = 0
    for lag in range(K - t_k - L_f):
        rec = engine(warning + [(s, W, -1, t_k + lag) for s in SYBILS], N_NODES, K, policy=policy, **kw)["records"]
        if t_k + lag + L_f >= t_k + A:
            outside += 1
            assert (rec["omega"][0], rec["pi"][0], rec["revoked"][0]) == (
                clean["omega"][0], clean["pi"][0], clean["revoked"][0]), lag
        else:
            inside += 1
            assert rec["revoked"][0] > clean["revoked"][0] + 1e-6, lag
    assert inside and outside


@pytest.mark.parametrize("lag", [13, 15, 20])
def test_no_commit_revokes_after_the_horizon_where_erlang_does_not(lag):
    t_k, K = 5, 40
    warning = [(W, T, -1, t_k)]
    attack = warning + [(s, W, -1, t_k + lag) for s in SYBILS]
    for policy in ("erlang", "no_commit"):
        clean = engine(warning, N_NODES, K, policy=policy)["records"]
        rec = engine(attack, N_NODES, K, policy=policy)["records"]
        if policy == "erlang":
            assert rec["revoked"][0] == clean["revoked"][0] and clean["revoked"][0] < TOL
        else:
            assert rec["revoked"][0] - clean["revoked"][0] > 0.1 and rec["omega"][0] == 0.0
            assert rec["pi"][0] == pytest.approx(1.0 - rec["revoked"][0], abs=TOL)   # pi is the survival S


def busy_stream():
    """Hazard on W at lags 0 to 3, same-pair repeats in one snapshot and later, and the history."""
    rows = HISTORY + [(W, T, -1, 6), (W, T, -1, 6), (W, T, -1, 8), (T, W, +1, 6)]
    rows += [(s, W, -1, 6 + lag) for lag in range(4) for s in SYBILS]
    return rows


def test_arrival_releases_every_record_at_arrival_and_revokes_nothing():
    rows = busy_stream()
    rec = engine(rows, N_NODES, policy="arrival")["records"]
    assert np.all(rec["omega"] == 1.0) and not rec["pi"].any() and not rec["revoked"].any() and rec["closed"].all()
    t_k = np.asarray(rows)[:, 3]
    assert np.all(rec["omega_by_snap"][t_k, np.arange(len(rows))] == 1.0)    # released in its own snapshot
    assert engine(rows, N_NODES)["records"]["revoked"].max() > 0.1          # erlang does revoke on this stream


@pytest.mark.parametrize("D", [4, 3])
def test_quarantine_commits_the_surviving_mass_at_D(D):
    t_k = 6
    rec = engine([(W, T, -1, t_k)] + [(s, W, -1, t_k) for s in SYBILS], N_NODES,
                 policy="quarantine", quarantine_D=D)["records"]
    lam = DE.witness_hazard([0.20] * 5)["lam"]      # live from age 1 (lag 0 + L_f) for H = 6 >= D - 1 snapshots
    ages = np.arange(21) + 1 - t_k                   # the record's age at the end of each snapshot
    omega, pi = rec["omega_by_snap"][:, 0], rec["pi_by_snap"][:, 0]
    assert np.all(omega[ages < D] == 0.0)            # nothing is released before age D
    survive = sum(p * math.exp(-x * (D - 1)) for p, x in zip(PSI_NEG, lam))
    assert np.all(np.abs(omega[ages >= D] - survive) < 1e-15) and rec["closed"][0]
    assert np.all(pi[ages >= D] == 0.0)
    held = (ages >= 1) & (ages < D)
    want = sum(p * np.exp(-x * (ages[held] - 1)) for p, x in zip(PSI_NEG, lam))
    assert np.all(np.abs(pi[held] - want) < 1e-15)   # escrowed and revocable until D: pi = S


def random_stream(seed=0, n=30, E=320, K=21):
    """Random ratings (30% negative) with 40 same-pair repeats, sorted by snapshot."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, n, E)
    v = (u + rng.integers(1, n, E)) % n
    snap = np.sort(rng.integers(0, K, E))
    sign = np.where(rng.random(E) < 0.3, -1, 1)
    first = rng.integers(0, E - 40, 40)
    later = np.minimum(snap[first] + rng.integers(0, 4, 40), K - 1)
    rows = np.column_stack([u, v, sign, snap])
    repeats = np.column_stack([u[first], v[first], sign[first], later])
    rows = np.concatenate([rows, repeats])
    return rows[np.argsort(rows[:, 3], kind="stable")], n


POLICY_GRID = [("erlang", {}), ("erlang", dict(N=1, phi=0.25)), ("erlang", dict(N=3, phi=0.75)),
               ("erlang", dict(N=5, phi=1.25)), ("arrival", {}), ("quarantine", {}),
               ("quarantine", dict(quarantine_D=1.5)), ("no_commit", {})]


@pytest.mark.parametrize("mb", [False, True])
@pytest.mark.parametrize("policy, kw", POLICY_GRID)
def test_mass_is_conserved_and_released_mass_is_monotone_for_every_policy(policy, kw, mb):
    rows, n = random_stream()
    levels = np.random.default_rng(1).integers(0, 4, size=(21, n)) if mb else None
    out = engine(rows, n, policy=policy, witness_cred=levels, **kw)
    rec = out["records"]
    assert np.max(np.abs(rec["omega"] + rec["pi"] + rec["revoked"] - 1.0)) <= TOL
    assert out["max_conservation_err"] <= TOL and out["monotone_verified"]
    assert np.all(np.diff(rec["omega_by_snap"], axis=0) >= -1e-15)
    for key in ("omega", "pi", "revoked_haz", "revoked_sup"):
        assert rec[key].min() >= -1e-15 and rec[key].max() <= 1.0 + 1e-15, key
    if policy == "arrival":
        assert np.all(rec["omega"] == 1.0)
    else:
        assert rec["revoked_haz"].max() > 0.05 and rec["revoked_sup"].max() > 0.05     # both losses occur
    if policy == "no_commit":
        assert not rec["omega_by_snap"].any() and not rec["closed"].any()


# ------------------------------------------------------------------ R57: no_commit reads retroactively
def test_no_commit_reads_the_surviving_weight_retroactively():
    """Ruling R57. Under no_commit the durable tallies behind kappa and V2 hold each received rating's surviving
    weight S, which records["pi"] reports, so no_commit differs from erlang in commitment only. kappa is positive
    wherever erlang's is, and it follows from the surviving-weight sums of each account's received ratings."""
    rows, n = random_stream()
    erlang, no_commit = engine(rows, n), engine(rows, n, policy="no_commit")
    assert (erlang["kappa_nodes"] > 0).sum() > 10
    assert np.all(no_commit["kappa_nodes"][erlang["kappa_nodes"] > 0] > 0)
    assert no_commit["max_conservation_err"] <= TOL and no_commit["monotone_verified"]
    survive, receiver, positive = no_commit["records"]["pi"], rows[:, 1], rows[:, 2] > 0
    pos = np.bincount(receiver[positive], weights=survive[positive], minlength=n)
    mass = pos + np.bincount(receiver[~positive], weights=survive[~positive], minlength=n)
    want = (pos + 1.0) / (mass + 2.0) * (1.0 - np.exp(-mass / 2.0))
    assert np.max(np.abs(no_commit["kappa_nodes"] - want)) < TOL


def test_no_commit_tallies_by_hand_on_a_toy():
    """One rating 0 -> 1 at snapshot 0 counts with its whole weight S = 1, so kappa_1 = (2/3)(1 - e^(-1/2)). When
    five witnesses then warn 0 at snapshot 2, the rating's survival S drops, and kappa_1 follows it."""
    lone = engine([(0, 1, +1, 0)], 2, policy="no_commit")
    assert lone["kappa_nodes"][1] == pytest.approx(2.0 / 3.0 * (1.0 - math.exp(-0.5)), abs=1e-15)
    warned = engine([(0, 1, +1, 0)] + [(s, 0, -1, 2) for s in range(2, 7)], 7, policy="no_commit")
    s = warned["records"]["pi"][0]
    assert s < 0.5
    assert warned["kappa_nodes"][1] == pytest.approx((s + 1.0) / (s + 2.0) * (1.0 - math.exp(-s / 2.0)), abs=1e-15)


# ------------------------------------------------------------------ 8. the SCEM module


def module_inputs(n=10, K=6, d=16, per_snapshot=6, seed=0):
    """Node features [n, K, d] and a rating stream of `per_snapshot` ratings per snapshot (30% negative)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, K, d, generator=gen)
    u = torch.randint(0, n, (K * per_snapshot,), generator=gen)
    v = (u + torch.randint(1, n, (K * per_snapshot,), generator=gen)) % n
    neg = torch.rand(K * per_snapshot, generator=gen) < 0.3
    labels = torch.stack([(~neg).float(), neg.float()], dim=1)
    ends = [(t + 1) * per_snapshot - 1 for t in range(K)]
    return x, torch.stack([u, v]), labels, ends


@pytest.fixture
def scem_module():
    torch.manual_seed(0)
    return SC.SCEM(input_dim=16, max_seq_len=6)


def run_module(module, *args, **kw):
    z = module(*args, **kw)
    return z.detach().clone(), float(module.last_aux_loss)


@pytest.mark.parametrize("mode", ["graph_stream", "sequence"])
def test_module_forward_is_unchanged_by_none_and_all_ones_witness_weights(scem_module, mode):
    x, edges, labels, ends = module_inputs()
    args = (x, edges, labels, ends) if mode == "graph_stream" else (x,)
    n, K = x.shape[:2]
    z0, a0 = run_module(scem_module, *args)
    z1, a1 = run_module(scem_module, *args, witness_weights=None)
    z2, a2 = run_module(scem_module, *args, witness_weights=torch.ones(K, n))
    assert torch.equal(z0, z1) and torch.equal(z0, z2) and a0 == a1 == a2
    z3, _ = run_module(scem_module, *args, witness_weights=torch.zeros(K, n))
    assert not torch.equal(z0, z3)                    # the weights do reach the witness term


def test_module_witness_weights_shape_is_checked(scem_module):
    x, edges, labels, ends = module_inputs()
    with pytest.raises(ValueError):
        scem_module(x, edges, labels, ends, witness_weights=torch.ones(5, 10))    # fewer rows than snapshots
    with pytest.raises(ValueError):
        scem_module(x, witness_weights=torch.ones(6, 9))                           # not one column per node


def test_witness_weights_shape_is_checked():
    WC.check_witness_weights(np.ones((22, 5)), 21, 5)
    for shape in ((20, 5), (21, 4), (21, 6), (21,)):
        with pytest.raises(ValueError):
            WC.check_witness_weights(np.ones(shape), 21, 5)


def test_witness_weighted_scem_keeps_a_private_copy_of_its_weights(scem_module):
    """A later change to the caller's array, numpy or tensor, must not alias into the fixed weights."""
    x, edges, labels, ends = module_inputs()
    n, K = x.shape[:2]
    want, _ = run_module(scem_module, x, edges, labels, ends, witness_weights=torch.ones(K, n))
    for weights in (np.ones((K, n)), torch.ones(K, n)):
        cls = stage_b.witness_weighted_scem(weights)
        weights[:] = 0.0                                   # the caller reuses its array
        torch.manual_seed(0)                               # the same initial parameters as scem_module
        got, _ = run_module(cls(input_dim=16, max_seq_len=6), x, edges, labels, ends)
        assert torch.equal(got, want)


@pytest.fixture
def pipeline_device(monkeypatch):
    """The device keeper.laundering.core chose; without a GPU the project's override allows the CPU."""
    if C.DEV.type == "cpu":
        monkeypatch.setenv(C.ALLOW_CPU_ENV, "1")
    return C.DEV.type


@pytest.mark.filterwarnings(r"ignore:`torch\.(cpu|cuda)\.amp\.autocast:FutureWarning")  # inside torch.utils.checkpoint
def test_all_ones_witness_weights_leave_the_archived_localgdte_unchanged(pipeline_device):
    D = C.load("bitcoinalpha")
    ones = stage_b.witness_weighted_scem(torch.ones(D.tt, D.n))
    out = stage_b.forward_check("bitcoinalpha", 0, pipeline_device, original_scem=ones)
    assert out["identical_initial_parameters"] and out["embeddings_equal"]
    assert out["max_abs_diff_embedding"] == 0.0 and out["aux_loss_original"] == out["aux_loss_patched"]
    zeros = stage_b.witness_weighted_scem(np.zeros((D.tt, D.n)))      # numpy, as level_weights returns them
    assert not stage_b.forward_check("bitcoinalpha", 0, pipeline_device, original_scem=zeros)["embeddings_equal"]
