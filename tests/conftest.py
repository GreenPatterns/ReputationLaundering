"""pytest configuration.

tests/test_core.py is the script-style GPU regression suite: its checks run at import time and need a
CUDA GPU, so a bare `pytest` on the CPU must not collect it (ruling R19). Run it on a GPU as
`python -m tests.test_core` from the repository root.

The stubs fixture replaces every model computation of the CPPR driver by a stand-in; tests/test_cp_vpe_driver.py and
tests/test_cp_vpe_base_cache.py share it (moved from the former in Task 4.1c fix round 1).
"""
collect_ignore = ["test_core.py"]

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def training_graphs():
    """ds -> (training graph, node count) of the three CPPR networks, loaded once (tests/credibility_common.py)."""
    from tests.credibility_common import load_training_graphs
    return load_training_graphs()


@pytest.fixture
def stubs(monkeypatch):
    """Replace every model computation by StubModel. Returns the list of training fingerprints, in call order."""
    import numpy as np

    from keeper.cppr import baselines as B
    from keeper.cppr import cp_vpe_gpu as G
    from keeper.laundering import core as C
    from tests import cp_vpe_legacy as O
    from tests.cppr_common import StubModel, fingerprint

    calls = []

    def train_gated_model(D, Tr, seed, tt, cred_levels, epochs=150, plain_vpe_anon_nodes=None,
                          plain_vpe_pool_weights=None, gate_norm="snapshot"):
        anon = None if plain_vpe_anon_nodes is None else np.asarray(plain_vpe_anon_nodes)
        calls.append(fingerprint("gated", D.name, int(D.n), int(D.T), Tr, np.asarray(cred_levels), seed, tt, epochs,
                                 anon, plain_vpe_pool_weights, gate_norm))
        return StubModel(Tr, cred_levels, seed, gate_norm, plain_vpe_pool_weights, plain_vpe_anon_nodes)

    def make_model(D, Tr, seed, tt, legacy_head=False, strategy=None):
        calls.append(fingerprint("ungated", D.name, int(D.n), Tr, seed, tt))
        return StubModel(Tr, np.zeros(Tr.n, dtype=np.int64), seed, "ungated")

    def train_withdrawal_model(D, Tr, seed, tt, cred_levels, epochs=150, gate_norm="snapshot", *, counts):
        calls.append(fingerprint("withdrawal", D.name, int(D.n), int(D.T), Tr, np.asarray(cred_levels), seed, tt,
                                 epochs, np.asarray(counts), gate_norm))
        return StubModel(Tr, cred_levels, seed, gate_norm)

    for module in (G, O):
        monkeypatch.setattr(module, "train_gated_model", train_gated_model)
    monkeypatch.setattr(B, "train_withdrawal_model", train_withdrawal_model)    # Task 4.3's withdrawal-count model
    monkeypatch.setattr(C, "make_model", make_model)
    monkeypatch.setattr(C, "train", lambda m, Tr, tt, epochs=150, lr=0.005, wd=1e-5: m)
    monkeypatch.setattr(C, "embed", lambda m, Tr, tt: m.z)
    monkeypatch.setattr(C, "require_gpu", lambda: None)
    monkeypatch.setattr(C, "set_deterministic", lambda on=True: None)
    monkeypatch.setattr(C, "apply_memory_profile", lambda D: {"stub": True})
    return calls
