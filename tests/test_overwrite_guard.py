"""Stored results are evidence: no writer replaces one without an explicit flag (ruling R27).

keeper.laundering.core.dump() and keeper.scem.stage_b.save() refuse to overwrite an existing output file
unless the caller passes force (the CLI's --force) or the environment sets KEEPER_OVERWRITE=1. Every
experiment CLI that writes a result also checks its output path before its work starts, so a refused run
stops at once instead of after hours of training. probe4's .partial.jsonl resume file is appended to and
never refused; only the final record is guarded. Derived aggregates (cp_vpe_gpu --aggregate, aggregate4,
stage_b aggregate) are regenerated only with the flag.

Every test writes into pytest's tmp_path: core.RES and stage_b.OUT are redirected there. No GPU needed.

Both writers are atomic (Task 4.1b fix round): a result appears complete or not at all, even when serialization
fails or another writer creates the same file meanwhile, and no temporary file is left behind.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from keeper.scem import stage_b  # first: it sets CUBLAS_WORKSPACE_CONFIG before CUDA starts
from keeper.laundering import core as C

ROOT = Path(__file__).resolve().parents[1]
STORED = '{"stored": true}'


class WorkStarted(Exception):
    """Raised by the first step of a writer's real work: reaching it means the guard let the run go on."""


def work_started(*args, **kwargs):
    raise WorkStarted


@pytest.fixture(autouse=True)
def no_overwrite_env(monkeypatch):
    monkeypatch.delenv(C.OVERWRITE_ENV, raising=False)


@pytest.fixture
def res(monkeypatch, tmp_path):
    """core.RES redirected to tmp_path, and the first step of every writer's work replaced by WorkStarted."""
    monkeypatch.setattr(C, "RES", str(tmp_path))
    monkeypatch.setattr(C, "set_deterministic", work_started)
    monkeypatch.setattr(C, "require_gpu", work_started)
    return tmp_path


@pytest.fixture
def out(monkeypatch, tmp_path):
    """stage_b.OUT redirected to tmp_path; the engine's data loader and the model build start the work."""
    monkeypatch.setattr(stage_b, "OUT", str(tmp_path))
    monkeypatch.setattr(stage_b, "load_stream", work_started)
    monkeypatch.setattr(stage_b, "scem_localgdte", work_started)
    return tmp_path


def test_results_live_under_results():
    from keeper.paperkit import cppr_gate_values, make_supp_tables
    assert C.RES == str(ROOT / "results" / "laundering")
    assert stage_b.OUT == str(ROOT / "results" / "scem" / "stage_b")
    assert make_supp_tables.DEFAULT_RESULTS == cppr_gate_values.RES == C.RES
    assert make_supp_tables.TABLE_DIR == str(ROOT / "results" / "paper" / "tables")
    assert make_supp_tables.VALUES_PATH == str(ROOT / "results" / "paper" / "values.json")
    assert os.path.isfile(os.path.join(C.RES, "calibration_bitcoinalpha_s0.json"))
    assert os.path.isfile(os.path.join(stage_b.OUT, "engine_bitcoinalpha.json"))


# ------------------------------------------------------------------ core.dump
def test_dump_writes_a_new_file(tmp_path):
    path = C.dump({"x": 1}, str(tmp_path / "r.json"))
    assert json.loads(Path(path).read_text())["x"] == 1


def test_dump_refuses_an_existing_file(tmp_path):
    target = tmp_path / "r.json"
    target.write_text(STORED)
    with pytest.raises(FileExistsError, match="exists; pass --force"):
        C.dump({"x": 1}, str(target))
    assert target.read_text() == STORED


@pytest.mark.parametrize("value", ["0", "", "yes", "true"])
def test_only_keeper_overwrite_1_unlocks(tmp_path, monkeypatch, value):
    target = tmp_path / "r.json"
    target.write_text(STORED)
    monkeypatch.setenv(C.OVERWRITE_ENV, value)
    with pytest.raises(FileExistsError):
        C.dump({"x": 1}, str(target))
    assert target.read_text() == STORED


def test_dump_overwrites_with_force_or_env(tmp_path, monkeypatch):
    target = tmp_path / "r.json"
    target.write_text(STORED)
    C.dump({"x": 1}, str(target), force=True)
    assert json.loads(target.read_text())["x"] == 1
    monkeypatch.setenv(C.OVERWRITE_ENV, "1")
    C.dump({"x": 2}, str(target))
    assert json.loads(target.read_text())["x"] == 2


def test_relative_names_resolve_in_RES(res):
    assert C.result_path("a.json") == str(res / "a.json")
    (res / "a.json").write_text(STORED)
    with pytest.raises(FileExistsError):
        C.dump({"x": 1}, "a.json")
    assert C.dump({"x": 1}, "a.json", force=True) == str(res / "a.json")


# ------------------------------------------------------------------ experiment CLIs
# (module, extra argv, default output file). Each main() must refuse an existing output before its work.
WRITERS = [
    ("keeper.laundering.calibrate_operators", [], "calibration_bitcoinalpha_s0.json"),
    ("keeper.laundering.incidental", [], "incidental_bitcoinalpha_s0.json"),
    ("keeper.laundering.probe_baselines", [], "baselines_bitcoinalpha_s0.json"),
    ("keeper.laundering.verification", [], "verification_bitcoinalpha_s0.json"),
    ("keeper.federated.fed_laundering", [], "fed_laundering_bitcoinalpha_s0.json"),
    ("keeper.cppr.vpe_frontier", [], "vpe_bitcoinalpha_s0.json"),
    ("keeper.laundering.probe4", [], "probe4_bitcoinalpha_s0.json"),
    ("keeper.laundering.aggregate4", [], "probe4_aggregate.json"),
    # Ruling G1 (Task 3.2): count is the default gate, so the snapshot ablation's runs need --gate-norm snapshot.
    ("keeper.cppr.cp_vpe_gpu", [], "cpgate_count_bitcoinalpha_s0.json"),
    ("keeper.cppr.cp_vpe_gpu", ["--gate-norm", "snapshot"], "cp_vpe_bitcoinalpha_s0.json"),
    ("keeper.cppr.cp_vpe_gpu", ["--gate-norm", "count"], "cpgate_count_bitcoinalpha_s0.json"),
]
IDS = [f"{m.rsplit('.', 1)[1]}{''.join(a)}" for m, a, _ in WRITERS]


def run_main(monkeypatch, module, argv):
    mod = importlib.import_module(module)
    if module.endswith("aggregate4"):
        monkeypatch.setattr(mod, "other_hypotheses", work_started)   # its first step reads the results
    monkeypatch.setattr(sys, "argv", [module, *argv])
    mod.main()


@pytest.mark.parametrize("module,argv,name", WRITERS, ids=IDS)
def test_cli_refuses_an_existing_result_before_its_work(res, monkeypatch, module, argv, name):
    (res / name).write_text(STORED)
    with pytest.raises(FileExistsError, match="exists; pass --force"):
        run_main(monkeypatch, module, argv)
    assert (res / name).read_text() == STORED


@pytest.mark.parametrize("module,argv,name", WRITERS, ids=IDS)
@pytest.mark.parametrize("unlock", ["absent", "--force", "env"])
def test_cli_starts_work_when_allowed(res, monkeypatch, module, argv, name, unlock):
    if unlock != "absent":
        (res / name).write_text(STORED)
    if unlock == "env":
        monkeypatch.setenv(C.OVERWRITE_ENV, "1")
    with pytest.raises(WorkStarted):
        run_main(monkeypatch, module, argv + (["--force"] if unlock == "--force" else []))
    assert unlock == "absent" or (res / name).read_text() == STORED   # nothing written before the work


def test_threshold_refuses_before_its_work(res):
    from keeper.laundering import threshold
    (res / "threshold_bitcoinalpha_s0.json").write_text(STORED)
    with pytest.raises(FileExistsError):
        threshold.pre_register("bitcoinalpha")
    with pytest.raises(WorkStarted):
        threshold.pre_register("bitcoinalpha", force=True)


def test_probe4_resume_file_does_not_block_a_run(res, monkeypatch):
    """A crashed run leaves only its .partial.jsonl: the rerun must go on and resume from it."""
    (res / "probe4_bitcoinalpha_s0.partial.jsonl").write_text('{"target": 1}\n')
    with pytest.raises(WorkStarted):
        run_main(monkeypatch, "keeper.laundering.probe4", [])
    (res / "probe4_bitcoinalpha_s0.json").write_text(STORED)      # the final record exists: refused
    with pytest.raises(FileExistsError):
        run_main(monkeypatch, "keeper.laundering.probe4", [])
    assert (res / "probe4_bitcoinalpha_s0.partial.jsonl").read_text() == '{"target": 1}\n'


# ------------------------------------------------------------------ derived aggregates
@pytest.mark.parametrize("argv,name", [(["--aggregate"], "cpgate_count_aggregate.json"),        # G1 default
                                       (["--aggregate", "--gate-norm", "snapshot"], "cp_vpe_gpu_aggregate.json"),
                                       (["--aggregate", "--gate-norm", "count"], "cpgate_count_aggregate.json")])
def test_cppr_aggregate_is_regenerated_only_with_the_flag(res, monkeypatch, argv, name):
    (res / name).write_text(STORED)                    # tmp_path holds no per-seed runs: the aggregate is {}
    with pytest.raises(FileExistsError):
        run_main(monkeypatch, "keeper.cppr.cp_vpe_gpu", argv)
    assert (res / name).read_text() == STORED
    run_main(monkeypatch, "keeper.cppr.cp_vpe_gpu", argv + ["--force"])
    assert json.loads((res / name).read_text()) == {}


# ------------------------------------------------------------------ stage_b
def run_stage_b(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["stage_b.py", *argv])
    stage_b.main()


@pytest.mark.parametrize("argv,name", [(["engine"], "engine_bitcoinalpha.json"),
                                       (["model"], "model_bitcoinalpha_s0.json"),
                                       (["trace"], "trace_symbolic.json"),
                                       (["aggregate"], "model_summary.json")])
def test_stage_b_refuses_an_existing_record(out, monkeypatch, capsys, argv, name):
    (out / name).write_text(STORED)
    with pytest.raises(SystemExit) as stop:
        run_stage_b(monkeypatch, *argv)
    assert stop.value.code == 2 and "exists; pass --force" in capsys.readouterr().err
    assert (out / name).read_text() == STORED


@pytest.mark.parametrize("unlock", ["--force", "env"])
def test_stage_b_starts_work_or_rewrites_when_allowed(out, monkeypatch, unlock):
    for name in ("engine_bitcoinalpha.json", "trace_symbolic.json", "model_summary.json"):
        (out / name).write_text(STORED)
    (out / "model_bitcoinalpha_s0.json").write_text('{"ds": "bitcoinalpha", "ordinary_mcc": 0.5}')
    flag = ["--force"] if unlock == "--force" else []
    if unlock == "env":
        monkeypatch.setenv(C.OVERWRITE_ENV, "1")
    for cmd in ("engine", "model"):                    # GPU or data work starts only after the check
        with pytest.raises(WorkStarted):
            run_stage_b(monkeypatch, cmd, *flag)
    run_stage_b(monkeypatch, "trace", *flag)           # cheap: the record is rewritten
    run_stage_b(monkeypatch, "aggregate", *flag)       # the summary of the one model record above
    assert json.loads((out / "trace_symbolic.json").read_text()) != json.loads(STORED)
    summary = json.loads((out / "model_summary.json").read_text())
    assert summary["bitcoinalpha"]["ordinary_mcc"] == {"mean": 0.5, "sd": None, "n": 1}


def test_stage_b_save_is_guarded(out):
    stage_b.save("r.json", {"x": 1})
    with pytest.raises(FileExistsError):
        stage_b.save("r.json", {"x": 2})
    stage_b.save("r.json", {"x": 3}, force=True)
    assert json.loads((out / "r.json").read_text()) == {"x": 3}


# ------------------------------------------------------------------ atomic writes (Task 4.1b fix round)
RECORD = {"x": float("nan"), "y": np.float32(1.5), "z": [np.int64(3)], "w": {"a": [1, 2.5, None]},
          "provenance": {"fixed": True}}          # a fixed provenance, so two writes give comparable bytes
UNSERIALIZABLE = {"a": 1, "b": object()}             # json.dump fails after it has written '{"a": 1, "b": '


def leftovers(folder) -> list:
    """Temporary files left in a folder."""
    return sorted(p.name for p in Path(folder).iterdir() if p.name.endswith(".tmp"))


def dump_before_the_fix(obj, path, D=None):
    """core.dump's write before the fix round, verbatim: the oracle of the bytes written."""
    obj = dict(obj)
    obj.setdefault("provenance", C.provenance(D))
    with open(path, "w") as f:
        json.dump(C._finite(obj), f, default=C._json_default, allow_nan=False)


def save_before_the_fix(path, obj):
    """stage_b.save's write before the fix round, verbatim."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def test_dump_writes_the_same_bytes_as_before(tmp_path):
    dump_before_the_fix(RECORD, str(tmp_path / "old.json"))
    C.dump(RECORD, str(tmp_path / "new.json"))
    assert (tmp_path / "new.json").read_bytes() == (tmp_path / "old.json").read_bytes()
    C.dump(RECORD, str(tmp_path / "old.json"), force=True)                 # a replacement too
    assert (tmp_path / "new.json").read_bytes() == (tmp_path / "old.json").read_bytes()
    assert leftovers(tmp_path) == []


def test_a_dump_that_fails_while_serializing_leaves_no_file(tmp_path):
    with pytest.raises(TypeError):
        C.dump(UNSERIALIZABLE, str(tmp_path / "r.json"))
    assert not (tmp_path / "r.json").exists() and leftovers(tmp_path) == []


def test_a_forced_dump_that_fails_keeps_the_stored_file(tmp_path):
    target = tmp_path / "r.json"
    target.write_text(STORED)
    with pytest.raises(TypeError):
        C.dump(UNSERIALIZABLE, str(target), force=True)
    assert target.read_text() == STORED and leftovers(tmp_path) == []


def test_dump_refuses_a_file_that_another_writer_creates_meanwhile(tmp_path, monkeypatch):
    # result_path checks first, so a refused run stops early; the final link refuses again, atomically.
    target = tmp_path / "r.json"
    checked = C.result_path

    def another_writer_finishes_first(name, force=False):
        path = checked(name, force)
        target.write_text(STORED)
        return path

    monkeypatch.setattr(C, "result_path", another_writer_finishes_first)
    with pytest.raises(FileExistsError):
        C.dump({"x": 1}, str(target))
    assert target.read_text() == STORED and leftovers(tmp_path) == []


def test_force_replaces_and_leaves_no_temporary(tmp_path, monkeypatch):
    target = tmp_path / "r.json"
    target.write_text(STORED)
    C.dump({"x": 1}, str(target), force=True)
    assert json.loads(target.read_text())["x"] == 1
    monkeypatch.setenv(C.OVERWRITE_ENV, "1")
    C.dump({"x": 2}, str(target))
    assert json.loads(target.read_text())["x"] == 2 and leftovers(tmp_path) == []


@pytest.mark.parametrize("replace", [False, True])
def test_write_atomic_fsyncs_the_complete_temporary_before_it_becomes_the_file(tmp_path, monkeypatch, replace):
    # The temporary is fsynced while it holds the whole text, and only then linked to the file (or moved over it), so
    # that a crash can leave the file absent but never partial (Task 4.1b review minor, done in Task 4.1c).
    target, events = tmp_path / "r.json", []
    real = dict(fsync=os.fsync, link=os.link, replace=os.replace)

    def fsync(fd):
        events.append(("fsync", [(p.name, p.read_text()) for p in tmp_path.iterdir() if p.name.endswith(".tmp")]))
        real["fsync"](fd)

    def spy(name):
        def call(src, dst, *args, **kwargs):
            events.append((name, os.path.basename(src), dst))
            return real[name](src, dst, *args, **kwargs)
        return call

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "link", spy("link"))
    monkeypatch.setattr(os, "replace", spy("replace"))
    if replace:
        target.write_text(STORED)
    C.write_atomic(str(target), lambda f: f.write('{"complete": true}'), replace=replace)
    (step, temporaries), (move, source, destination) = events
    assert step == "fsync" and move == ("replace" if replace else "link") and destination == str(target)
    assert temporaries == [(source, '{"complete": true}')]            # the whole text, in the file that moves
    assert target.read_text() == '{"complete": true}' and leftovers(tmp_path) == []


def test_write_atomic_writes_bytes_in_binary_mode(tmp_path):
    # binary=True opens the temporary in binary mode, for the base-model cache's entries (Task 4.1c fix round 1). The
    # overwrite guard, the replacement and the cleanup are those of a text write.
    target, data = tmp_path / "entry.pt", bytes(range(256)) * 3
    C.write_atomic(str(target), lambda f: f.write(data), binary=True)
    assert target.read_bytes() == data and leftovers(tmp_path) == []
    with pytest.raises(FileExistsError):
        C.write_atomic(str(target), lambda f: f.write(b"other"), binary=True)
    assert target.read_bytes() == data and leftovers(tmp_path) == []
    C.write_atomic(str(target), lambda f: f.write(b"other"), replace=True, binary=True)
    assert target.read_bytes() == b"other" and leftovers(tmp_path) == []


def test_stage_b_save_writes_the_same_bytes_as_before(out):
    save_before_the_fix(str(out / "old.json"), RECORD)
    stage_b.save("new.json", RECORD)
    assert (out / "new.json").read_bytes() == (out / "old.json").read_bytes()
    stage_b.save("old.json", RECORD, force=True)
    assert (out / "new.json").read_bytes() == (out / "old.json").read_bytes() and leftovers(out) == []


def test_a_stage_b_save_that_fails_leaves_no_file(out):
    with pytest.raises(TypeError):
        stage_b.save("r.json", UNSERIALIZABLE)
    assert not (out / "r.json").exists() and leftovers(out) == []
    (out / "r.json").write_text(STORED)
    with pytest.raises(TypeError):
        stage_b.save("r.json", UNSERIALIZABLE, force=True)
    assert (out / "r.json").read_text() == STORED and leftovers(out) == []


def test_stage_b_save_refuses_a_record_that_another_writer_creates_meanwhile(out, monkeypatch):
    checked = stage_b.record_path

    def another_writer_finishes_first(name, force=False):
        path = checked(name, force)
        (out / name).write_text(STORED)
        return path

    monkeypatch.setattr(stage_b, "record_path", another_writer_finishes_first)
    with pytest.raises(FileExistsError):
        stage_b.save("r.json", {"x": 1})
    assert (out / "r.json").read_text() == STORED and leftovers(out) == []
    stage_b.save("r.json", {"x": 2}, force=True)                           # force replaces
    assert json.loads((out / "r.json").read_text()) == {"x": 2} and leftovers(out) == []
