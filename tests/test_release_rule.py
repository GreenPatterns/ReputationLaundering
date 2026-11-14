"""Release-rule ablation (paper Table III, Block 1) regenerated from the one SCEM engine.

keeper.scem.release_rule applies the release script's ex-post rule to every engine configuration of
Stage B, with the engine keeper.scem.scem.run_scem_streaming_weights_np. The first test is the STOP
gate of Task 1.4 (ruling R3): the default engine (N = 2, phi = 0.5) must give the published
attenuation 63.278% / 66.242% / 69.148% within 5e-5. The second test regenerates the whole stored
output and requires it byte for byte. CPU only (the engine is NumPy); about a minute.
"""
from pathlib import Path

from keeper.scem import release_rule

STORED = (Path(__file__).resolve().parents[1]
          / "results" / "scem" / "release_rule" / "release_rule_ablation_exact.txt")


def test_release_rule_attenuation_reproduced():
    got = {ds: release_rule.attenuation_default_rule(ds) for ds in ("bitcoinalpha", "bitcoinotc", "wiki-rfa")}
    assert abs(got["bitcoinalpha"] - 0.63278) < 5e-5
    assert abs(got["bitcoinotc"] - 0.66242) < 5e-5
    assert abs(got["wiki-rfa"] - 0.69148) < 5e-5


def test_release_rule_output_is_byte_identical_to_stored_file(capsys):
    release_rule.main()
    assert capsys.readouterr().out.encode() == STORED.read_bytes()
