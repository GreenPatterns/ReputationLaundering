"""CLI glue for rerun checks (Task 1.6); no computation changes.

probe4 --only-target: a rerun check processes one target of the same draw. It needs --out, so a
one-target record never takes a full run's file name. Without the flag the CLI is unchanged.
stage_b --out: records go to another folder. The trace written there is byte-identical to the stored
record, and the stored record is untouched.

CPU only; every write goes to pytest's tmp_path.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from keeper.scem import stage_b  # first: it sets CUBLAS_WORKSPACE_CONFIG before CUDA starts
from keeper.laundering import core as C
from keeper.laundering import probe4

DRAW = np.array([3509, 3510, 3522], dtype=np.int64)


def parse(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["probe4.py", *args])
    return probe4.parse()


def test_select_targets_keeps_the_whole_draw_by_default():
    assert probe4.select_targets(DRAW) is DRAW


def test_select_targets_picks_one_target_of_the_draw():
    tg = probe4.select_targets(DRAW, 3510)
    assert tg.tolist() == [3510] and tg.dtype == np.int64


def test_select_targets_refuses_a_target_outside_the_draw():
    with pytest.raises(SystemExit, match="not in the draw"):
        probe4.select_targets(DRAW, 1)


def test_only_target_needs_out(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        parse(monkeypatch, "--only-target", "3509")
    assert stop.value.code == 2 and "--only-target needs --out" in capsys.readouterr().err


def test_only_target_with_out_parses(monkeypatch, tmp_path):
    a = parse(monkeypatch, "--only-target", "3509", "--out", str(tmp_path / "r.json"))
    assert a.only_target == 3509 and a.out == str(tmp_path / "r.json")


def test_without_only_target_the_cli_is_unchanged(monkeypatch):
    a = parse(monkeypatch)
    assert a.only_target is None and a.out is None


def test_stage_b_out_writes_the_record_to_the_given_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(stage_b, "OUT", stage_b.OUT)          # main() rebinds it; restored after the test
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)
    stored = Path(stage_b.OUT) / "trace_symbolic.json"
    before = stored.read_bytes()
    monkeypatch.setattr(sys, "argv", ["stage_b.py", "trace", "--out", str(tmp_path)])
    stage_b.main()
    assert stage_b.OUT == str(tmp_path)
    assert (tmp_path / "trace_symbolic.json").read_bytes() == before
    assert stored.read_bytes() == before
