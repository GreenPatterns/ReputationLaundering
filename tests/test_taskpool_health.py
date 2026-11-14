"""keeper.cluster.taskpool: a faulty GPU or node, and failures that are not the task's own. CPU only, no GPU.

A retry runs on a different GPU than the failed attempt when the pool has another one, and waits for it rather than
reuse the failed one. A GPU is dropped after two failures in a row on it, or after a failure shorter than
--min-runtime seconds. After --max-consecutive-failures failures in a row, on any GPUs, or when no GPU is left, the
pool stops: it lets what runs finish, releases its claims, leaves the rest of the task list untouched and exits 3.
A launch error (a log folder that cannot be made) is a failed attempt, not a crash.

The tests run with FAST settings (no drops, no halt, see tests/taskpool_common.py) unless they are about exactly those.
"""
import errno

from keeper.cluster import taskpool
from tests import taskpool_common as common
from tests.taskpool_common import LivePool, counts, wait_until

ws, run = common.ws, common.run


# ------------------------------------------------------------------ retries on another GPU
def test_a_retry_runs_on_a_different_gpu_than_the_failed_attempt(ws, run):
    ws.write(ws.line("t", "failon", "3"))                          # fails on GPU 3 only
    res = run()
    assert res.code == 0 and res.summary == counts(ran_ok=1)
    assert [(r["attempt"], r["gpu"], r["outcome"]) for r in ws.status()] == [(1, "3", "retry"), (2, "5", "ok")]


def test_a_retry_waits_for_a_different_gpu_instead_of_reusing_the_failed_one(ws):
    release = ws.root / "release"
    ws.write(ws.line("t", "failon", "3"), ws.line("gate", "gate", str(release)), ws.line("u"))
    with LivePool(ws, "--slots-per-gpu", "1") as pool:             # t runs on GPU 3, gate on GPU 5, u waits for a slot
        wait_until(lambda: ws.attempts("u"), "u to run on the GPU that t left")   # its status line, not its done file
        assert ws.attempts("t") == [(1, "retry")] and len(ws.ran("t")) == 1     # the retry waits: GPU 5 is busy
        assert [r["gpu"] for r in ws.status() if r["id"] == "u"] == ["3"]       # the freed GPU went to another task
        release.write_text("go")
        assert pool.wait() == 0
    assert [(r["attempt"], r["gpu"]) for r in ws.status() if r["id"] == "t"] == [(1, "3"), (2, "5")]
    assert ws.done("t").exists()


def test_a_pool_with_one_gpu_retries_on_it(ws, run):
    ws.write(ws.line("t", "seq", "fail,ok"))
    res = run(gpus="3")
    assert res.code == 0 and [(r["attempt"], r["gpu"]) for r in ws.status()] == [(1, "3"), (2, "3")]


# ------------------------------------------------------------------ dropping a GPU
def test_a_gpu_is_dropped_after_two_failures_in_a_row_on_it(ws):
    release = ws.root / "release"
    ws.write(ws.line("f0", "failon", "3"), ws.line("gate", "gate", str(release)), ws.line("f1", "failon", "3"),
             *(ws.line(i, "failon", "3") for i in ("f2", "a", "b")))
    with LivePool(ws, "--retries", "0", "--slots-per-gpu", "1") as pool:    # GPU 3 takes f0 and f1, GPU 5 the gate
        wait_until(lambda: [r["id"] for r in ws.status()] == ["f0", "f1"], "f0 and f1 to fail on GPU 3")
        release.write_text("go")
        assert pool.wait() == 1                                              # two tasks failed
        assert "dropping GPU 3" in pool.text
    gpus = {r["id"]: r["gpu"] for r in ws.status()}
    assert gpus == {"f0": "3", "f1": "3", "gate": "5", "f2": "5", "a": "5", "b": "5"}   # nothing on 3 after the second
    assert [r["outcome"] for r in ws.status() if r["gpu"] == "3"] == ["failed", "failed"]


def test_a_gpu_is_dropped_after_a_failure_shorter_than_min_runtime(ws, run):
    ws.write(ws.line("f", "failon", "3"), *(ws.line(f"ok{i}") for i in range(3)))
    res = run("--min-runtime", "3600", "--slots-per-gpu", "1")      # every failure here is shorter than an hour
    rows = ws.status()
    assert [r["id"] for r in rows if r["gpu"] == "3"] == ["f"]     # nothing ran on GPU 3 after the quick failure
    assert res.code == 0 and ws.done("f").exists() and "dropping GPU 3" in res.out


def test_gpu_health_counts_failures_per_gpu_and_in_a_row_across_gpus():
    health = taskpool.GpuHealth(("3", "5"), min_runtime=10, max_consecutive=3)
    health.note("3", False, 20)
    health.note("5", True, 20)                                     # a success resets the run of failures ...
    assert health.usable() == ["3", "5"] and health.streak == 0
    health.note("3", False, 20)
    health.note("3", False, 20)                                    # ... but not the strikes of a GPU without a success
    assert health.usable() == ["5"] and health.streak == 2 and health.halted is None
    health.note("5", False, 5)                                     # a failure shorter than min_runtime drops its GPU
    assert health.usable() == [] and health.streak == 3 and health.halted
    health.note("5", True, 99)                                     # a success on a dropped GPU does not bring it back
    assert health.usable() == [] and health.streak == 0


def test_a_success_on_a_gpu_clears_its_strikes_and_a_failure_that_lasted_min_runtime_keeps_it():
    health = taskpool.GpuHealth(("3", "5"), min_runtime=10, max_consecutive=99)
    health.note("3", False, 20)
    health.note("3", True, 20)
    health.note("3", False, 20)
    assert health.usable() == ["3", "5"]                           # two failures, but not in a row
    health.note("5", False, 10)
    assert health.usable() == ["3", "5"]                           # 10 s is not shorter than --min-runtime 10


def test_a_launch_that_failed_counts_in_the_run_of_failures_but_not_against_a_gpu():
    health = taskpool.GpuHealth(("3",), min_runtime=10, max_consecutive=2)
    health.note(None, False, 0)
    health.note(None, False, 0)
    assert health.usable() == ["3"] and health.halted


# ------------------------------------------------------------------ halting the pool
def test_the_pool_stops_after_max_consecutive_failures_and_leaves_the_rest_untouched(ws, run):
    ids = [f"t{i:02d}" for i in range(12)]
    ws.write(*(ws.line(i, "fail") for i in ids))
    res = run("--retries", "0", "--max-consecutive-failures", "3", "--slots-per-gpu", "1", gpus="3,5,7")
    assert res.code == 3
    tried = [r["id"] for r in ws.status()]
    assert 3 <= len(tried) <= 6 and res.summary["failed"] == len(tried)     # three in a row, plus what was running
    untouched = [i for i in ids if not ws.log(i).exists() and not ws.claim(i).exists()]
    assert len(untouched) >= 6 and not set(untouched) & set(tried)
    assert not list((ws.root / "out").glob("*.claim"))
    assert "3 failures in a row" in res.out


def test_a_halted_pool_releases_the_retry_it_held_and_the_next_pool_carries_on(ws, run):
    ws.write(ws.line("t", "seq", "fail,ok"), ws.line("u"))
    res = run("--max-consecutive-failures", "1", "--slots-per-gpu", "1", gpus="3")
    assert res.code == 3 and not ws.claim("t").exists()
    assert ws.attempts("t") == [(1, "retry")] and not ws.log("u").exists()   # t's budget is on record, u untouched
    later = run()
    assert later.code == 0 and later.summary == counts(ran_ok=2)
    assert ws.attempts("t") == [(1, "retry"), (2, "ok")]                      # the budget carried over


def test_a_halted_pool_waits_for_the_tasks_still_running_and_records_them(ws):
    release = ws.root / "release"
    ws.write(ws.line("slow", "gate", str(release)), ws.line("bad", "fail"), ws.line("never"))
    with LivePool(ws, "--retries", "0", "--max-consecutive-failures", "1", "--slots-per-gpu", "1") as pool:
        wait_until(lambda: ws.attempts("bad") == [(1, "failed")], "bad to fail and halt the pool")
        assert pool.proc.poll() is None and ws.claim("slow").exists()      # the pool waits for slow to finish
        release.write_text("go")
        assert pool.wait() == 3
    assert ws.attempts("slow") == [(1, "ok")] and not ws.claim("slow").exists()
    assert not ws.log("never").exists() and "stopped: 1 failures in a row" in pool.text


def test_with_every_gpu_dropped_the_pool_stops_and_leaves_the_rest_untouched(ws, run):
    ws.write(ws.line("f", "fail"), ws.line("a"), ws.line("b"))
    res = run("--min-runtime", "3600", "--slots-per-gpu", "1", gpus="3")
    assert res.code == 3 and "no usable GPU" in res.out
    assert ws.attempts("f") == [(1, "retry")] and not ws.claim("f").exists()
    assert ws.ran("a") == [] and ws.ran("b") == []


# ------------------------------------------------------------------ launch errors
def test_a_launch_error_is_a_failed_attempt_not_a_crash(ws, run, monkeypatch):
    ws.write(ws.line("broken"), ws.line("fine"))
    real_launch = taskpool.launch

    def launch_unless_broken(task, gpu, cwd, header):
        if task.id == "broken":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_launch(task, gpu, cwd, header)

    monkeypatch.setattr(taskpool, "launch", launch_unless_broken)
    res = run("--min-runtime", "3600")                             # an instant failure would drop a GPU, but not this
    assert res.code == 1 and res.summary == counts(ran_ok=1, failed=1)
    rows = [r for r in ws.status() if r["id"] == "broken"]
    assert [(r["attempt"], r["outcome"], r["exit_code"]) for r in rows] == [(1, "retry", None), (2, "failed", None)]
    assert "No space left" in rows[0]["error"] and "dropping GPU" not in res.out
    assert not ws.claim("broken").exists() and ws.done("fine").exists()
