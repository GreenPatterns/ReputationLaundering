"""keeper.cluster.taskpool: the GPU task-pool runner of Phase 4 (Task 4.1a). CPU only: fake GPU ids, dummy commands.

Success and retries, the child's environment, logs and working directory, scheduling, the status file, signals and
cleanup, and the dry run. The claims, the status history that keeps the retry budget per task list and the task
file are in tests/test_taskpool_state.py; GPU health, the halt and the launch errors in
tests/test_taskpool_health.py. The helpers (the dummy task, Workspace, LivePool) are in tests/taskpool_common.py.
"""
import contextlib
import errno
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from keeper.cluster import taskpool, taskpool_process, taskpool_state
from tests import taskpool_common as common
from tests.taskpool_common import (FAST, SUMMARY, UTC, LivePool, counts, is_running, max_overlap, parse_summary,
                                   wait_until)

ws, run, dead_pid = common.ws, common.run, common.dead_pid


# ------------------------------------------------------------------ done files and success
def test_a_task_whose_done_file_exists_is_skipped_and_counted_done(ws, run):
    ws.write(ws.line("old"), ws.line("new"))
    ws.done("old").parent.mkdir(parents=True)
    ws.done("old").write_text("PRE")
    res = run()
    assert res.code == 0
    assert ws.done("old").read_text() == "PRE" and ws.ran("old") == []
    assert len(ws.ran("new")) == 1 and ws.done("new").exists() and ws.log("new").exists()   # log folder created
    assert not ws.log("old").exists()                                  # a skipped task gets no log, claim or status
    assert res.summary == counts(done=1, ran_ok=1)
    assert [row["id"] for row in ws.status()] == ["new"]


@pytest.mark.parametrize("mode, exit_code", [("nodone", 0), ("donefail", 1)])
def test_success_needs_exit_zero_and_the_done_file_together(ws, run, mode, exit_code):
    ws.write(ws.line("good"), ws.line("bad", mode))
    res = run("--retries", "0")
    assert res.code == 1
    assert res.summary == counts(ran_ok=1, failed=1)
    rows = {row["id"]: row for row in ws.status()}
    assert (rows["good"]["outcome"], rows["bad"]["outcome"], rows["bad"]["exit_code"]) == ("ok", "failed", exit_code)


# ------------------------------------------------------------------ retries
def test_a_failing_task_is_retried_once_then_recorded_failed(ws, run):
    ws.write(ws.line("t", "fail"))
    res = run()                                       # --retries defaults to 1, the preregistered rule
    assert res.code == 1 and res.summary["failed"] == 1
    assert [(row["attempt"], row["outcome"], row["exit_code"]) for row in ws.status()] == [(1, "retry", 3),
                                                                                           (2, "failed", 3)]
    assert all("claim=True" in entry for entry in ws.ran("t")) and len(ws.ran("t")) == 2   # claim kept in between
    assert not ws.claim("t").exists()
    assert "failed tasks: t" in res.out


@pytest.mark.parametrize("retries", [0, 1, 3])
def test_retries_sets_how_often_a_failing_task_is_rerun(ws, run, retries):
    ws.write(ws.line("t", "fail"))
    run("--retries", str(retries))
    assert len(ws.ran("t")) == retries + 1
    assert [row["outcome"] for row in ws.status()] == ["retry"] * retries + ["failed"]


def test_a_retry_that_succeeds_counts_as_ran_ok(ws, run):
    ws.write(ws.line("t", "seq", "fail,ok"))
    res = run()
    assert res.code == 0 and res.summary == counts(ran_ok=1)
    assert [row["outcome"] for row in ws.status()] == ["retry", "ok"]
    assert ws.done("t").exists() and not ws.claim("t").exists()


# ------------------------------------------------------------------ GPUs and the child's environment
def test_each_task_sees_exactly_one_gpu_and_the_cublas_setting(ws, run, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,5,6")      # the pool's own allocation: replaced per task
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "bogus")
    monkeypatch.setenv("KEEPER_TEST_MARK", "inherited")
    ids = [f"t{i}" for i in range(6)]
    ws.write(*(ws.line(i) for i in ids))
    assert run(gpus="3,5").code == 0
    seen = [ws.result(i) for i in ids]
    assert {r["CUDA_VISIBLE_DEVICES"] for r in seen} == {"3", "5"}
    assert all(r["CUBLAS_WORKSPACE_CONFIG"] == taskpool.CUBLAS_WORKSPACE_CONFIG == ":4096:8" for r in seen)
    assert all(r["KEEPER_TEST_MARK"] == "inherited" for r in seen)
    assert {row["gpu"] for row in ws.status()} == {"3", "5"}


def test_gpus_default_to_cuda_visible_devices(ws, run, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7, 9")
    ws.write(*(ws.line(f"t{i}") for i in range(4)))
    assert run(gpus=None).code == 0
    assert {ws.result(f"t{i}")["CUDA_VISIBLE_DEVICES"] for i in range(4)} == {"7", "9"}


@pytest.mark.parametrize("cuda_visible_devices, gpus", [(None, None), ("", None), (" , ", None), ("0,1", "")])
def test_refuses_to_start_without_a_gpu(ws, run, monkeypatch, cuda_visible_devices, gpus):
    if cuda_visible_devices is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    ws.write(ws.line("t"))
    for extra in ([], ["--dry-run"]):                                  # a dry run needs the GPUs to plan the slots
        res = run(*extra, gpus=gpus)
        assert res.code == 2 and "GPU" in res.err
    assert ws.ran("t") == [] and not ws.claim("t").exists() and ws.status() == []


# ------------------------------------------------------------------ scheduling
def test_concurrency_never_exceeds_slots_per_gpu_times_gpus_and_reaches_it(ws, run):
    n_tasks, record = 10, ws.root / "record"
    ws.write(*(ws.line(f"t{i}", "barrier", str(record), "4", "0.2") for i in range(n_tasks)))
    res = run("--slots-per-gpu", "2", "--retries", "0", gpus="3,5")
    assert res.code == 0 and res.summary["ran_ok"] == n_tasks
    spans = [json.loads(path.read_text()) for path in record.glob("span_*.json")]
    assert len(spans) == n_tasks
    assert max_overlap(spans) == 4                    # 2 GPUs x 2 slots: forced by the barrier, never exceeded
    for gpu in ("3", "5"):
        assert max_overlap([s for s in spans if s["gpu"] == gpu]) == 2


def test_tasks_start_in_task_file_order_and_a_retry_comes_before_the_later_tasks(ws, run):
    ws.write(ws.line("t0", "seq", "fail,ok"), *(ws.line(f"t{i}") for i in range(1, 5)))
    run("--slots-per-gpu", "1", gpus="3")                              # one slot: the tasks run one after the other
    assert [(row["id"], row["attempt"]) for row in ws.status()] == [
        ("t0", 1), ("t0", 2), ("t1", 1), ("t2", 1), ("t3", 1), ("t4", 1)]


def test_tasks_are_spread_over_the_gpus_before_any_gpu_gets_a_second_one(ws, run):
    record = ws.root / "record"
    ws.write(*(ws.line(f"t{i}", "barrier", str(record), "2", "0") for i in range(2)))
    run("--slots-per-gpu", "2", "--retries", "0", gpus="3,5")        # both start together: the barrier needs two
    assert {row["gpu"] for row in ws.status()} == {"3", "5"}


@pytest.mark.parametrize("flag, bad", [("--slots-per-gpu", "0"), ("--slots-per-gpu", "x"), ("--poll", "0"),
                                       ("--poll", "-1"), ("--poll", "x"), ("--retries", "-1"),
                                       ("--min-runtime", "-1"), ("--max-consecutive-failures", "0")])
def test_the_numeric_options_must_be_sensible(ws, run, flag, bad):
    ws.write(ws.line("t"))
    with pytest.raises(SystemExit) as stop:
        run(flag, bad)
    assert stop.value.code == 2 and ws.ran("t") == []


def test_the_defaults_of_the_documented_options():
    args = taskpool.build_parser().parse_args(["tasks.tsv"])
    assert (args.slots_per_gpu, args.retries, args.poll, args.min_runtime, args.max_consecutive_failures) == (
        2, 1, 5.0, 60.0, 4)
    assert not (args.steal_stale or args.rerun_failed or args.dry_run)


# ------------------------------------------------------------------ logs, cwd, shell
def test_log_gets_a_header_per_attempt_and_the_output_of_both_streams(ws, run):
    ws.write(ws.line("t", "fail"))
    ws.log("t").parent.mkdir(parents=True)
    ws.log("t").write_text("OLD\n")
    run()
    text = ws.log("t").read_text()
    headers = [line for line in text.splitlines() if line.startswith("[taskpool]")]
    assert text.startswith("OLD\n") and len(headers) == 2             # appended, one header per attempt
    assert "task=t " in headers[0] and "attempt=1 " in headers[0] and "attempt=2 " in headers[1]
    assert f"host={socket.gethostname()} " in headers[0]
    assert re.search(r" gpu=(3|5) utc=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", headers[0])
    first, second = text.index(headers[0]), text.index(headers[1])
    assert text.count("boom on stdout") == 2 and text.count("boom on stderr") == 2
    assert first < text.index("boom on stdout") < second < text.rindex("boom on stdout")


def test_the_command_is_a_shell_command_run_by_bash_in_the_cwd(ws, run):
    work = ws.root / "work"
    work.mkdir()
    command = f'[[ "$(basename "$PWD")" == work ]] && echo "$((6 * 7))" > {ws.done("t")}'     # bashisms
    ws.write(ws.line("t", command=command))
    assert run("--cwd", str(work)).code == 0
    assert ws.done("t").read_text().strip() == "42"


def test_cwd_defaults_to_the_current_directory(ws, run, monkeypatch):
    work = ws.root / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    ws.write(ws.line("t"))
    assert run(cwd=False).code == 0
    assert Path(ws.result("t")["cwd"]).resolve() == work.resolve()


def test_a_missing_cwd_is_refused(ws, run):
    ws.write(ws.line("t"))
    res = run("--cwd", str(ws.root / "nowhere"))
    assert res.code == 2 and "nowhere" in res.err and ws.ran("t") == []


# ------------------------------------------------------------------ status file
def test_a_pool_is_named_by_slurm_job_host_and_pid(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert taskpool.pool_id("h") == f"h:{os.getpid()}"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert taskpool.pool_id("h") == f"slurm123:h:{os.getpid()}"


def test_status_lines_record_each_attempt_with_the_documented_fields(ws, run):
    ws.write(ws.line("t"))
    run()
    (row,) = ws.status()
    assert list(row) == ["id", "attempt", "host", "pool", "gpu", "start_utc", "end_utc", "seconds", "exit_code",
                         "outcome"]
    assert (row["id"], row["attempt"], row["exit_code"], row["outcome"]) == ("t", 1, 0, "ok")
    assert row["host"] == socket.gethostname() and row["gpu"] in ("3", "5")
    assert row["pool"].endswith(f"{socket.gethostname()}:{os.getpid()}")     # the pool of the claim, per attempt
    assert UTC.fullmatch(row["start_utc"]) and UTC.fullmatch(row["end_utc"]) and row["start_utc"] <= row["end_utc"]
    assert isinstance(row["seconds"], float) and row["seconds"] >= 0


def test_status_file_default_path_custom_path_and_appending(ws, run):
    ws.write(ws.line("a"))
    run()
    assert [row["id"] for row in ws.status()] == ["a"]                  # ws.status() reads <task list>.status.jsonl
    ws.write(ws.line("a"), ws.line("b"))
    run()
    assert [row["id"] for row in ws.status()] == ["a", "b"]             # appended to, not replaced
    custom = ws.root / "deep" / "er" / "s.jsonl"
    ws.write(ws.line("c"))
    run("--status", str(custom))
    assert [row["id"] for row in ws.status(custom)] == ["c"]            # folders created; the default file untouched
    assert [row["id"] for row in ws.status()] == ["a", "b"]


def test_a_status_line_is_written_while_the_exclusive_flock_is_held(tmp_path, monkeypatch):
    events, real_flock, real_open = [], fcntl.flock, open

    class Recorded:                                  # the status file, with its writes and its close on record
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            events.append("close")
            self.f.close()

        def fileno(self):
            return self.f.fileno()

        def write(self, text):
            events.append("write")
            return self.f.write(text)

        def flush(self):
            self.f.flush()

    monkeypatch.setattr(taskpool_state, "open", lambda *a, **k: Recorded(real_open(*a, **k)), raising=False)
    monkeypatch.setattr(fcntl, "flock", lambda f, op: (events.append(("flock", op)), real_flock(f, op))[1])
    taskpool_state.append_status(str(tmp_path / "status.jsonl"), {"id": "a"})
    assert events == [("flock", fcntl.LOCK_EX), "write", "close"]       # locked before the write, held until the close
    assert (tmp_path / "status.jsonl").read_text() == '{"id": "a"}\n'


# ------------------------------------------------------------------ several pools
def test_several_pools_on_one_task_list_never_run_a_task_twice(ws):
    ids = [f"t{i:02d}" for i in range(12)]
    ws.write(*(ws.line(i, "sleep", "0.15") for i in ids))
    with contextlib.ExitStack() as stack:
        pools = [stack.enter_context(LivePool(ws, "--slots-per-gpu", "1", name=f"pool{k}")) for k in range(3)]
        codes = [pool.wait() for pool in pools]
    summaries = [parse_summary(pool.text) for pool in pools]
    assert codes == [0, 0, 0]
    assert [len(ws.ran(i)) for i in ids] == [1] * len(ids)
    assert all(ws.done(i).exists() for i in ids)
    assert sum(s["ran_ok"] for s in summaries) == len(ids)
    assert all(sum(s.values()) == len(ids) for s in summaries)         # every task is in exactly one bucket
    rows = ws.status()
    assert sorted(row["id"] for row in rows) == ids and {row["outcome"] for row in rows} == {"ok"}
    assert all(any(row["pool"].endswith(f":{pool.proc.pid}") for pool in pools) for row in rows)   # names its pool
    assert not list((ws.root / "out").glob("*.claim"))


# ------------------------------------------------------------------ signals and cleanup
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_signal_terminates_the_running_children_then_releases_their_claims(ws, sig):
    ids = ["t1", "t2", "t3"]
    ws.write(*(ws.line(i, "hold") for i in ids))
    with LivePool(ws, "--slots-per-gpu", "1") as pool:                 # 2 GPUs x 1 slot: t1 and t2 run, t3 waits
        wait_until(lambda: ws.ran("t1") and ws.ran("t2"), "t1 and t2 to start")
        info = json.loads(ws.claim("t1").read_text())
        assert set(info) == {"host", "pid", "pgid", "pool", "utc"}
        assert info["pid"] == pool.proc.pid and info["host"] == socket.gethostname() and UTC.fullmatch(info["utc"])
        assert not ws.claim("t3").exists()                            # a task is claimed only when it starts
        pids = [ws.child_pid("t1"), ws.child_pid("t2")]
        pool.proc.send_signal(sig)
        code = pool.wait()
    assert code == 128 + sig
    assert not any(ws.claim(i).exists() for i in ids)
    assert not any(is_running(pid) for pid in pids)
    assert ws.ran("t3") == [] and not any(ws.done(i).exists() for i in ids)
    assert ws.status() == []                                           # an interrupted attempt is not an outcome
    assert SUMMARY.search(pool.text)


def test_termination_reaches_processes_a_command_started_in_the_background(ws):
    pidfile, started = ws.root / "bg.pid", ws.root / "started"
    ws.write(ws.line("t", command=f"sleep 120 & echo $! > {pidfile}; echo up > {started}; wait"))
    background = None
    try:
        with LivePool(ws, "--slots-per-gpu", "1", gpus="3") as pool:
            wait_until(started.exists, "the command to start")
            background = int(pidfile.read_text())
            assert is_running(background)
            pool.proc.send_signal(signal.SIGTERM)
            assert pool.wait() == 128 + signal.SIGTERM
        wait_until(lambda: not is_running(background), "the background process to end")
    finally:
        if background:
            with contextlib.suppress(ProcessLookupError):
                os.kill(background, signal.SIGKILL)
    assert not ws.claim("t").exists()


def test_terminate_all_escalates_to_sigkill_for_a_group_that_ignores_sigterm(tmp_path):
    ready = tmp_path / "ready"
    proc = subprocess.Popen(["bash", "-c", f"trap '' TERM; touch {ready}; while :; do sleep 0.1; done"],
                            start_new_session=True)
    try:
        wait_until(ready.exists, "the shell to ignore SIGTERM")
        ender = threading.Thread(target=taskpool.terminate_all, args=([proc],), kwargs=dict(grace=0.3))
        ender.start()
        ender.join(timeout=30)
        assert not ender.is_alive(), "terminate_all keeps waiting for a process that ignores SIGTERM"
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            common.kill_group(proc.pid)                       # the shell and the sleep inside its loop
            proc.wait()


def test_terminate_all_kills_a_member_that_ignores_sigterm_after_the_leader_is_gone(tmp_path):
    pidfile = tmp_path / "member.pid"
    leader = subprocess.Popen(["bash", "-c", f"(trap '' TERM; echo $BASHPID > {pidfile}; while :; do sleep 0.1; done)"
                                             " & wait"], start_new_session=True)
    member = None
    try:
        wait_until(lambda: pidfile.exists() and pidfile.read_text().strip(), "the member to ignore SIGTERM")
        member = int(pidfile.read_text())
        ender = threading.Thread(target=taskpool.terminate_all, args=([leader],), kwargs=dict(grace=0.5))
        ender.start()
        ender.join(timeout=30)
        assert not ender.is_alive()
        assert leader.returncode == -signal.SIGTERM               # the leader obeyed at once ...
        assert not is_running(member)                             # ... the member needed the SIGKILL to the group
    finally:
        common.kill_group(leader.pid)
        leader.wait()


def test_terminate_all_gives_the_members_of_a_group_their_grace_after_the_leader_is_gone(tmp_path):
    ready, marker = tmp_path / "ready", tmp_path / "marker"
    member = f"(trap 'sleep 0.3; echo done > {marker}; exit 0' TERM; echo up > {ready}; while :; do sleep 0.1; done)"
    leader = subprocess.Popen(["bash", "-c", member + " & wait"], start_new_session=True)
    try:
        wait_until(ready.exists, "the member to be ready for SIGTERM")
        taskpool.terminate_all([leader], grace=10)
        assert marker.exists()                          # SIGTERM reached the member, which was left to finish by itself
    finally:
        common.kill_group(leader.pid)
        leader.wait()


def test_claims_are_released_only_after_the_children_are_dead(ws, monkeypatch):
    ready = ws.root / "ready"
    monkeypatch.setattr(taskpool_process, "TERM_GRACE_SECONDS", 1.0)
    ws.write(ws.line("t", command=f"trap '' TERM; echo $$ > {ready}; while :; do sleep 0.1; done"))
    pool = taskpool.Pool(ws.settings(gpus=("3",), retries=0))
    result, child = [], None
    thread = threading.Thread(target=lambda: result.append(pool.run()))
    thread.start()
    try:
        wait_until(lambda: ready.exists() and ready.read_text().strip(), "the child to ignore SIGTERM")
        child = int(ready.read_text())
        assert ws.claim("t").exists()
        pool.request_stop(signal.SIGTERM)
        wait_until(lambda: not ws.claim("t").exists(), "the claim to be released")
        assert not is_running(child)                                   # the child must already be dead by then
    finally:
        if child:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child, signal.SIGKILL)
        thread.join(timeout=60)
    assert result == [128 + signal.SIGTERM]


def test_an_error_in_the_pool_still_ends_the_children_and_releases_the_claims(ws, monkeypatch):
    ws.write(ws.line("t1", "hold"), ws.line("t2"))

    def disk_full(path, record):
        wait_until(lambda: ws.ran("t1"), "t1 to start")                # so that its pid is on record
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(taskpool, "append_status", disk_full)
    with pytest.raises(OSError, match="No space"):
        taskpool.main([str(ws.tsv), *FAST, "--gpus", "3", "--slots-per-gpu", "2", "--cwd", str(ws.root)])
    assert not ws.claim("t1").exists() and not ws.claim("t2").exists()
    assert not is_running(ws.child_pid("t1"))


def test_no_task_is_launched_after_a_stop_is_requested(ws, monkeypatch):
    ws.write(*(ws.line(f"t{i}", "hold") for i in range(4)))
    pool = taskpool.Pool(ws.settings(slots=2))                         # four free slots at the start
    real_launch = taskpool.launch

    def launch_then_stop(*args):
        proc = real_launch(*args)
        pool.request_stop(signal.SIGTERM)                              # the stop arrives while the slots are filled
        return proc

    monkeypatch.setattr(taskpool, "launch", launch_then_stop)
    assert pool.run() == 128 + signal.SIGTERM
    assert sum(ws.log(f"t{i}").exists() for i in range(4)) == 1       # a log exists only for a launched task
    assert not list((ws.root / "out").glob("*.claim"))


def test_shutdown_gives_a_child_that_had_already_exited_its_status_line(ws, monkeypatch):
    ws.write(ws.line("crash", "fail"), ws.line("busy", "hold"))
    pool = taskpool.Pool(ws.settings())
    real_launch, procs = taskpool.launch, {}

    def launch_then_stop_once_the_crash_is_over(task, gpu, cwd, header):
        procs[task.id] = real_launch(task, gpu, cwd, header)
        if task.id == "busy":                                          # "crash" was launched first and fails at once
            wait_until(lambda: procs["crash"].poll() is not None, "the crash to be over")
            pool.request_stop(signal.SIGTERM)
        return procs[task.id]

    monkeypatch.setattr(taskpool, "launch", launch_then_stop_once_the_crash_is_over)
    assert pool.run() == 128 + signal.SIGTERM
    assert [(row["id"], row["exit_code"], row["outcome"]) for row in ws.status()] == [("crash", 3, "retry")]
    assert not list((ws.root / "out").glob("*.claim"))                # the retry's claim is released with the rest


def test_signal_handlers_are_restored_after_a_run(ws, run):
    ws.write(ws.line("t"))
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    before = [signal.getsignal(s) for s in signals]
    assert run().code == 0
    assert [signal.getsignal(s) for s in signals] == before


# ------------------------------------------------------------------ dry run
def test_dry_run_prints_the_plan_and_runs_nothing(ws, run):
    ws.write(ws.line("finished"), ws.line("held"), ws.line("p1"), ws.line("p2"))
    ws.done("finished").parent.mkdir(parents=True)
    ws.done("finished").write_text("PRE")
    ws.put_claim("held", common.OWNER)
    res = run("--dry-run", "--slots-per-gpu", "3")
    assert res.code == 0
    assert "= 6 slots" in res.out and "3,5" in res.out                # 2 GPUs x 3 slots
    assert "1 done" in res.out and "1 claimed by another pool" in res.out and "2 pending" in res.out
    assert re.findall(r"^  (\S+)  ", res.out, re.M) == ["p1", "p2"]    # pending, in TSV order
    for task_id in ("held", "p1", "p2"):
        assert ws.ran(task_id) == [] and not ws.done(task_id).exists()
    assert not ws.claim("p1").exists() and not ws.log("p1").parent.exists() and ws.status() == []
    assert not SUMMARY.search(res.out)


def test_dry_run_reports_missing_tasks_and_counts_a_stale_claim_as_pending_only_with_steal_stale(ws, run, dead_pid):
    ws.write(ws.line("spent"), ws.line("t"))
    ws.put_status(dict(id="spent", attempt=1, outcome="retry"), dict(id="spent", attempt=2, outcome="failed"))
    ws.put_claim("t", dict(common.OWNER, pid=dead_pid))
    plain = run("--dry-run").out
    assert "1 claimed by another pool" in plain and "1 missing" in plain and "= 4 slots" in plain   # default 2 slots
    res = run("--dry-run", "--steal-stale")
    assert "1 pending" in res.out and ws.claim("t").exists()          # the dry run does not release it
    assert "0 missing" in run("--dry-run", "--rerun-failed").out        # a rerun would try the spent task again
