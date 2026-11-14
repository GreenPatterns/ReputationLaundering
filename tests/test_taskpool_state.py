"""keeper.cluster.taskpool_state and the task file: claims, stale claims, the retry budget kept per task list, done
files that failed attempts leave behind, and the checks before anything starts. CPU only, nothing touches a GPU.

The budget lives in the shared status file: whoever claims a task reads the attempts that every pool recorded for it
(under a shared flock), skips it as `missing` when the budget is spent, and otherwise continues the numbering.
`--rerun-failed` is the deliberate override. Helpers: tests/taskpool_common.py.
"""
import contextlib
import errno
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from keeper.cluster import taskpool, taskpool_state
from tests import taskpool_common as common
from tests.taskpool_common import OWNER, LivePool, counts, is_running, parse_summary, wait_until

ws, run, dead_pid = common.ws, common.run, common.dead_pid
HOST = socket.gethostname()


def row(attempt, outcome, task="t", **extra) -> dict:
    return dict(id=task, attempt=attempt, outcome=outcome, **extra)


# ------------------------------------------------------------------ claims
def test_a_task_with_an_existing_claim_is_skipped_and_its_claim_is_left(ws, run):
    ws.write(ws.line("held"), ws.line("free"))
    ws.put_claim("held", OWNER)
    before = ws.claim("held").read_text()
    res = run()
    assert res.code == 0
    assert ws.ran("held") == [] and len(ws.ran("free")) == 1
    assert ws.claim("held").read_text() == before
    assert res.summary == counts(ran_ok=1, skipped_claimed=1)
    assert "claimed by another pool: held" in res.out


def test_try_claim_is_exclusive_and_records_the_owner(tmp_path):
    path = str(tmp_path / "not" / "yet" / "x.json.claim")             # its folder does not exist yet
    owner = dict(host="h", pid=1, pgid=None, pool="p", utc="u")
    assert taskpool.try_claim(path, owner) is True
    assert taskpool.try_claim(path, dict(owner, pool="q")) is False
    assert json.loads(Path(path).read_text()) == owner


def test_release_claim_removes_only_a_claim_the_pool_owns(tmp_path):
    mine, theirs = tmp_path / "mine.claim", tmp_path / "theirs.claim"
    taskpool.try_claim(str(mine), dict(host="h", pid=1, pool="me", utc="u"))
    taskpool.try_claim(str(theirs), dict(host="h", pid=2, pool="someone else", utc="u"))
    taskpool.release_claim(str(mine), "me")
    taskpool.release_claim(str(theirs), "me")
    taskpool.release_claim(str(tmp_path / "gone.claim"), "me")        # nothing there: not an error
    assert not mine.exists() and theirs.exists()


def test_set_claim_pgid_changes_only_the_pgid_of_a_claim_the_pool_owns(tmp_path):
    mine, theirs = tmp_path / "mine.claim", tmp_path / "theirs.claim"
    owner = dict(host="h", pid=1, pgid=None, pool="me", utc="u")
    taskpool.try_claim(str(mine), owner)
    taskpool.try_claim(str(theirs), dict(owner, pool="someone else"))
    taskpool.set_claim_pgid(str(mine), "me", 4321)
    taskpool.set_claim_pgid(str(theirs), "me", 4321)
    taskpool.set_claim_pgid(str(tmp_path / "gone.claim"), "me", 4321)  # nothing there: nothing is created
    assert json.loads(mine.read_text()) == dict(owner, pgid=4321)
    assert json.loads(theirs.read_text())["pgid"] is None and sorted(p.name for p in tmp_path.iterdir()) == [
        "mine.claim", "theirs.claim"]


def test_the_claim_names_the_process_group_of_the_running_attempt(ws):
    ws.write(ws.line("t", "hold"))
    with LivePool(ws, "--slots-per-gpu", "1", gpus="3") as pool:
        wait_until(lambda: ws.ran("t"), "the task to start")
        pid = ws.child_pid("t")
        wait_until(lambda: json.loads(ws.claim("t").read_text())["pgid"] == pid, "the claim to name the child's group")
        assert os.getpgid(pid) == pid                                   # the child leads its own group
        pool.proc.send_signal(signal.SIGTERM)
        pool.wait()


def test_a_finished_tasks_claim_is_released_at_once_not_when_the_pool_ends(ws):
    release = ws.root / "release"
    ws.write(ws.line("quick"), ws.line("slow", "gate", str(release)))
    with LivePool(ws, "--slots-per-gpu", "2", gpus="3") as pool:
        wait_until(lambda: ws.done("quick").exists() and not ws.claim("quick").exists(), "quick's claim to go")
        assert ws.claim("slow").exists() and pool.proc.poll() is None        # slow runs on, holding its claim
        release.write_text("go")
        assert pool.wait() == 0
    assert ws.done("slow").exists()


def test_every_status_line_is_written_while_the_claim_is_still_held(ws, run, monkeypatch):
    ws.write(ws.line("ok"), ws.line("bad", "fail"))
    real, seen = taskpool.append_status, []

    def spy(path, record):
        seen.append((record["id"], record["outcome"], ws.claim(record["id"]).exists()))
        return real(path, record)

    monkeypatch.setattr(taskpool, "append_status", spy)
    run()
    assert [(i, o) for i, o, _ in seen] and all(held for _, _, held in seen)    # released only after the line is in


def test_a_claim_that_cannot_be_written_is_not_left_behind(tmp_path, monkeypatch):
    def full_disk(*args, **kwargs):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(taskpool_state.json, "dump", full_disk)
    with pytest.raises(OSError, match="quota"):
        taskpool.try_claim(str(tmp_path / "x.json.claim"), dict(host="h"))
    assert list(tmp_path.iterdir()) == []                               # no empty claim to block the task for good


def test_a_task_finished_by_another_pool_between_the_check_and_the_claim_is_not_rerun(ws, run, monkeypatch):
    ws.write(ws.line("t"))
    real = taskpool.try_claim

    def claim_just_after_another_pool_finished(path, owner):
        ws.done("t").parent.mkdir(parents=True, exist_ok=True)
        ws.done("t").write_text("by another pool")
        return real(path, owner)

    monkeypatch.setattr(taskpool, "try_claim", claim_just_after_another_pool_finished)
    res = run()
    assert ws.ran("t") == [] and ws.done("t").read_text() == "by another pool"
    assert not ws.claim("t").exists()
    assert res.summary == counts(done=1)


def test_a_claimed_task_is_looked_at_once_and_never_taken_over_later(ws, run, monkeypatch):
    """A task that another pool gave up on and released must not get further runs from this pool."""
    ws.write(ws.line("held"), ws.line("free"))
    ws.put_claim("held", OWNER)
    real_claim, found_taken = taskpool.try_claim, []

    def release_after_finding_it_taken(path, owner):
        taken = real_claim(path, owner)
        if not taken and path.endswith("held.json.claim"):
            found_taken.append(path)
            Path(path).unlink()                                       # the other pool lets go right after
        return taken

    monkeypatch.setattr(taskpool, "try_claim", release_after_finding_it_taken)
    res = run("--slots-per-gpu", "1", gpus="3")                       # one slot: "held" is looked at before "free"
    assert found_taken and ws.ran("held") == []
    assert res.summary == counts(ran_ok=1, skipped_claimed=1)


# ------------------------------------------------------------------ stale claims
def test_steal_stale_takes_over_a_claim_of_a_dead_process_on_this_host(ws, run, dead_pid):
    ws.write(ws.line("t"))
    ws.put_claim("t", dict(OWNER, pid=dead_pid))
    res = run("--steal-stale")
    assert res.code == 0 and res.summary["ran_ok"] == 1
    assert ws.done("t").exists() and not ws.claim("t").exists()
    assert "claim=True" in ws.ran("t")[0]                              # the thief's own claim, held while running


@pytest.mark.parametrize("case", ["dead owner without the flag", "live owner", "other host", "unreadable claim"])
def test_a_claim_that_is_not_provably_stale_is_left_alone(ws, run, dead_pid, case):
    owners = {"dead owner without the flag": dict(OWNER, pid=dead_pid), "live owner": OWNER,
              "other host": dict(OWNER, host="another-host", pid=dead_pid), "unreadable claim": "{partial"}
    ws.write(ws.line("t"))
    ws.put_claim("t", owners[case])
    before = ws.claim("t").read_text()
    flags = [] if case == "dead owner without the flag" else ["--steal-stale"]
    res = run(*flags)
    assert res.code == 0 and res.summary["skipped_claimed"] == 1
    assert ws.ran("t") == [] and ws.claim("t").read_text() == before


def test_steal_stale_waits_until_the_process_group_of_the_dead_pools_attempt_has_ended(ws, run, dead_pid):
    orphan = subprocess.Popen(["sleep", "120"], start_new_session=True)       # an attempt that outlived its pool
    try:
        ws.write(ws.line("t"))
        ws.put_claim("t", dict(OWNER, pid=dead_pid, pgid=orphan.pid))
        res = run("--steal-stale")
        assert res.summary == counts(skipped_claimed=1) and ws.ran("t") == []   # its group is still alive
        orphan.kill()
        orphan.wait()
        assert run("--steal-stale").summary == counts(ran_ok=1)                 # now the claim is stale
    finally:
        common.kill_group(orphan.pid)


@pytest.mark.parametrize("claim", [
    dict(host="h", pid=0), dict(host="h", pid=-5), dict(host="h", pid="abc"), dict(host="h", pid=True),
    dict(host="h"), dict(pid=1), dict(host="h", pid=os.getpid()), dict(host="other", pid=common.DEAD_PID),
    dict(host="h", pid=common.DEAD_PID, pgid=os.getpgrp())])
def test_a_claim_that_is_unclear_or_may_have_a_live_owner_is_not_stale(claim):
    assert taskpool.is_stale(claim, "h") is False


@pytest.mark.parametrize("claim", [dict(host="h", pid=common.DEAD_PID), dict(host="h", pid=common.DEAD_PID, pgid=None),
                                   dict(host="h", pid=common.DEAD_PID, pgid=common.DEAD_PID)])
def test_a_claim_of_a_dead_pid_on_this_host_whose_group_is_gone_is_stale(claim):
    assert taskpool.is_stale(claim, "h") is True


def test_a_zombie_is_not_alive():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])   # exits at once and stays unreaped until wait()
    try:
        wait_until(lambda: not is_running(proc.pid), "the child to exit")
        assert os.path.exists(f"/proc/{proc.pid}")                    # still in the process table, as a zombie
        assert not taskpool.pid_alive(proc.pid) and taskpool.pid_alive(os.getpid())
    finally:
        proc.wait()


def test_a_steal_takes_the_flock_of_the_claim_itself_and_removes_a_stale_claim(tmp_path, monkeypatch):
    path = tmp_path / "x.json.claim"
    path.write_text(json.dumps(dict(OWNER, pid=common.DEAD_PID)))
    inode, locked, real = path.stat().st_ino, [], fcntl.flock
    monkeypatch.setattr(fcntl, "flock",
                        lambda f, op: (locked.append((os.fstat(f.fileno()).st_ino, op)), real(f, op))[1])
    assert taskpool.steal_stale(str(path), HOST) is True
    assert locked == [(inode, fcntl.LOCK_EX)] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("how", ["a new file replaces it", "the file is rewritten in place"])
def test_a_claim_taken_over_while_the_thief_waited_for_the_lock_is_left_alone(tmp_path, monkeypatch, how):
    path = tmp_path / "x.json.claim"
    path.write_text(json.dumps(dict(OWNER, pid=common.DEAD_PID)))
    live, real = dict(OWNER, pool="the live pool"), fcntl.flock

    def another_pool_claims_first(f, op):
        if how.startswith("a new file"):
            (tmp_path / "live.tmp").write_text(json.dumps(live))
            os.replace(tmp_path / "live.tmp", path)
        else:
            path.write_text(json.dumps(live))
        return real(f, op)

    monkeypatch.setattr(fcntl, "flock", another_pool_claims_first)
    assert taskpool.steal_stale(str(path), HOST) is False
    monkeypatch.undo()
    assert json.loads(path.read_text()) == live


def test_a_claim_removed_while_the_thief_waited_for_the_lock_leaves_the_way_free(tmp_path, monkeypatch):
    path = tmp_path / "x.json.claim"
    path.write_text(json.dumps(dict(OWNER, pid=common.DEAD_PID)))
    real = fcntl.flock

    def another_pool_removes_it(f, op):
        path.unlink()
        return real(f, op)

    monkeypatch.setattr(fcntl, "flock", another_pool_removes_it)
    assert taskpool.steal_stale(str(path), HOST) is True


def test_stealing_a_claim_that_is_already_gone_leaves_the_way_free(tmp_path):
    assert taskpool.steal_stale(str(tmp_path / "gone.claim"), HOST) is True
    (tmp_path / "x.claim").write_text("{partial")
    assert taskpool.steal_stale(str(tmp_path / "x.claim"), HOST) is False   # unreadable: left alone


def test_pools_stealing_the_same_stale_claims_at_once_run_each_task_once(ws, dead_pid):
    ids = [f"t{i:02d}" for i in range(10)]
    ws.write(*(ws.line(i, "sleep", "0.1") for i in ids))
    for task_id in ids:
        ws.put_claim(task_id, dict(OWNER, pid=dead_pid))
    with contextlib.ExitStack() as stack:
        pools = [stack.enter_context(LivePool(ws, "--slots-per-gpu", "2", "--steal-stale", name=f"pool{k}"))
                 for k in range(3)]
        codes = [pool.wait() for pool in pools]
    assert codes == [0, 0, 0]
    assert [len(ws.ran(i)) for i in ids] == [1] * len(ids) and all(ws.done(i).exists() for i in ids)
    assert not list((ws.root / "out").glob("*.claim*"))                # no claim and no trace of a steal


# ------------------------------------------------------------------ the retry budget kept per task list
@pytest.mark.parametrize("rows, expected", [
    ([], []),
    ([(1, "retry")], [(1, "retry")]),
    ([(1, "retry"), (2, "failed")], [(1, "retry"), (2, "failed")]),
    ([(1, "retry"), (2, "ok")], []),                                           # a success closes the budget
    ([(1, "ok")], []),
    ([(1, "retry"), (2, "failed"), (1, "retry")], [(1, "retry")]),             # a rerun starts a new budget
    ([(1, "retry"), (2, "failed"), (1, "retry"), (2, "failed")], [(1, "retry"), (2, "failed")]),
    ([(1, "retry"), (2, "ok"), (1, "retry")], [(1, "retry")]),
    ([(1, "failed"), (1, "ok")], [])])
def test_current_attempts_are_those_since_the_last_success_or_fresh_start(rows, expected):
    given = [row(attempt, outcome) for attempt, outcome in rows]
    assert [(r["attempt"], r["outcome"]) for r in taskpool.current_attempts(given)] == expected


@pytest.mark.parametrize("attempts, retries, spent", [
    ([], 1, False), ([(1, "retry")], 1, False), ([(1, "retry")], 0, True), ([(1, "failed")], 3, True),
    ([(1, "retry"), (2, "retry")], 1, True), ([(1, "retry"), (2, "retry")], 2, False),
    ([(1, "retry"), (2, "retry"), (3, "retry")], 2, True)])
def test_the_budget_is_spent_by_a_failed_line_or_by_retries_plus_one_attempts(attempts, retries, spent):
    assert taskpool.budget_spent([row(a, o) for a, o in attempts], retries) is spent


def test_the_history_collects_the_attempts_of_one_task_from_every_pool(ws):
    ws.put_status(row(1, "retry", pool="pool A"), row(1, "ok", task="other"), row(2, "failed", pool="pool B"),
                  row(2, "missing"))
    with open(ws.status_path, "a") as f:
        f.write("not json\n\n")
    rows = taskpool.read_history(str(ws.status_path), "t")
    assert [(r["attempt"], r["outcome"], r["pool"]) for r in rows] == [(1, "retry", "pool A"), (2, "failed", "pool B")]
    assert taskpool.read_history(str(ws.root / "none.jsonl"), "t") == []


def test_the_history_is_read_under_a_shared_lock(ws, monkeypatch):
    ws.put_status(row(1, "retry"))
    ops, real = [], fcntl.flock
    monkeypatch.setattr(fcntl, "flock", lambda f, op: (ops.append(op), real(f, op))[1])
    taskpool.read_history(str(ws.status_path), "t")
    assert ops == [fcntl.LOCK_SH]


def test_a_task_that_spent_its_budget_is_not_started_again_by_a_later_run(ws, run):
    ws.write(ws.line("bad", "fail"), ws.line("good"))
    first = run()
    assert first.code == 1 and first.summary == counts(ran_ok=1, failed=1) and len(ws.ran("bad")) == 2
    for _ in range(2):                                                # also after a run that wrote a `missing` line
        later = run()
        assert len(ws.ran("bad")) == 2                                # not started again
        assert later.code == 0 and later.summary == counts(done=1, missing=1)
        assert "missing tasks: bad" in later.out and not ws.claim("bad").exists()
    missing = [r for r in ws.status() if r["outcome"] == "missing"]
    assert len(missing) == 2 and missing[0]["attempt"] == 2 and missing[0]["exit_code"] is None
    assert all(r["pool"] for r in missing) and missing[0]["gpu"] is None


def test_a_second_live_pool_does_not_start_a_task_the_first_has_failed(ws):
    release = ws.root / "release"
    ws.write(ws.line("bad", "fail"), ws.line("slow", "gate", str(release)))
    with LivePool(ws, "--slots-per-gpu", "2", name="first") as first:
        wait_until(lambda: ws.attempts("bad") == [(1, "retry"), (2, "failed")], "bad to fail twice")
        with LivePool(ws, name="second") as second:                   # started while the first pool is still alive
            assert second.wait() == 0
        assert parse_summary(second.text) == counts(skipped_claimed=1, missing=1)
        assert len(ws.ran("bad")) == 2
        release.write_text("go")
        assert first.wait() == 1                                      # the first pool failed "bad"
    assert ws.done("slow").exists()


def test_a_recorded_retry_followed_by_an_interrupt_leaves_exactly_one_more_attempt(ws, run):
    ws.write(ws.line("t", "seq", "fail,hold,fail"))
    with LivePool(ws, name="first") as first:
        wait_until(lambda: len(ws.ran("t")) == 2, "attempt 2 to start")     # attempt 1 failed and is on record
        first.proc.send_signal(signal.SIGTERM)
        assert first.wait() == 128 + signal.SIGTERM
    assert ws.attempts("t") == [(1, "retry")]                          # the interrupted attempt 2 left no line
    res = run()                                                        # the next pool: one attempt is left
    assert res.code == 1 and len(ws.ran("t")) == 3
    assert ws.attempts("t") == [(1, "retry"), (2, "failed")]


def test_a_retry_on_record_from_an_earlier_pool_leaves_one_attempt(ws, run):
    ws.write(ws.line("t", "fail"))
    ws.put_status(row(1, "retry", pool="an earlier pool"))
    res = run()
    assert res.code == 1 and len(ws.ran("t")) == 1
    assert ws.attempts("t") == [(1, "retry"), (2, "failed")]
    assert "attempt=2 " in ws.log("t").read_text()


def test_a_success_on_record_gives_a_task_whose_result_was_removed_a_fresh_budget(ws, run):
    ws.write(ws.line("t", "fail"))
    ws.put_status(row(1, "retry"), row(2, "ok"))                       # it succeeded once; its result is gone
    res = run()
    assert res.summary == counts(failed=1) and len(ws.ran("t")) == 2   # a full budget of two attempts
    assert ws.attempts("t")[2:] == [(1, "retry"), (2, "failed")]


def test_rerun_failed_ignores_the_history_and_says_so_in_the_status(ws, run):
    ws.write(ws.line("t", "seq", "fail,fail,ok"))
    run()                                                              # the budget is spent after two attempts
    assert ws.attempts("t") == [(1, "retry"), (2, "failed")]
    assert run().summary == counts(missing=1)                          # an ordinary run leaves the task alone
    res = run("--rerun-failed")
    assert res.code == 0 and res.summary == counts(ran_ok=1) and len(ws.ran("t")) == 3
    last = ws.status()[-1]
    assert (last["attempt"], last["outcome"], last["rerun_failed"]) == (1, "ok", True)
    assert all("rerun_failed" not in r for r in ws.status()[:-1])      # only the lines of the rerun carry the flag


def test_a_rerun_that_fails_again_is_not_run_a_third_time_by_an_ordinary_pool(ws, run):
    ws.write(ws.line("t", "fail"))
    ws.put_status(row(1, "retry"), row(2, "failed"))
    run("--rerun-failed")
    assert ws.attempts("t")[2:] == [(1, "retry"), (2, "failed")]       # a fresh budget, spent again
    assert run().summary == counts(missing=1) and len(ws.ran("t")) == 2


def test_a_rerun_that_was_cut_short_is_continued_by_an_ordinary_pool(ws, run):
    ws.write(ws.line("t", "fail"))
    ws.put_status(row(1, "retry"), row(2, "failed"), row(1, "retry", rerun_failed=True))
    res = run()                                                        # one attempt of the rerun's budget is left
    assert res.code == 1 and len(ws.ran("t")) == 1 and ws.attempts("t")[3:] == [(2, "failed")]


# ------------------------------------------------------------------ done files that failed attempts leave behind
def test_a_done_file_left_by_a_failed_attempt_is_set_aside_not_counted_done(ws, run):
    ws.write(ws.line("t", "donefail"))
    res = run()
    assert res.code == 1 and res.summary == counts(failed=1)
    assert not ws.done("t").exists()                                   # no later pool can count it as done
    aside = ws.aside("t")
    assert [p.name for p in aside] == ["t.json.failed-a1", "t.json.failed-a2"]
    assert all(json.loads(p.read_text())["cwd"] for p in aside)        # the files are kept whole, as evidence
    assert [r["done_moved_to"] for r in ws.status()] == [str(p) for p in aside]
    later = run()
    assert later.summary == counts(missing=1) and len(ws.ran("t")) == 2


def test_a_retry_that_exits_zero_is_not_ok_because_the_failed_attempt_left_a_file(ws, run):
    ws.write(ws.line("t", "seq", "donefail,nodone"))
    res = run()
    assert ws.attempts("t") == [(1, "retry"), (2, "failed")]           # attempt 2 exited 0 but wrote nothing
    assert res.code == 1 and not ws.done("t").exists()
    assert [r["exit_code"] for r in ws.status()] == [1, 0]


def test_a_done_file_with_a_failed_attempt_on_record_behind_it_is_set_aside_and_the_task_goes_on(ws, run):
    ws.write(ws.line("t"))
    ws.put_status(row(1, "retry"))                                     # attempt 1 failed ...
    ws.done("t").parent.mkdir(parents=True)
    ws.done("t").write_text("TRUNCATED")                               # ... and left this behind
    res = run()
    assert res.summary == counts(ran_ok=1) and ws.attempts("t") == [(1, "retry"), (2, "ok")]
    assert [p.read_text() for p in ws.aside("t")] == ["TRUNCATED"]
    assert json.loads(ws.done("t").read_text())["cwd"]                 # the new, complete file


def test_a_done_file_after_a_success_on_record_counts_as_done(ws, run):
    ws.write(ws.line("t"))
    ws.put_status(row(1, "retry"), row(2, "ok"))
    ws.done("t").parent.mkdir(parents=True)
    ws.done("t").write_text("RESULT")
    assert run().summary == counts(done=1) and ws.ran("t") == [] and ws.aside("t") == []


def test_rerun_failed_counts_a_done_file_as_done_whatever_the_history_says(ws, run):
    ws.write(ws.line("t"))
    ws.put_status(row(1, "retry"))
    ws.done("t").parent.mkdir(parents=True)
    ws.done("t").write_text("RESULT")
    assert run("--rerun-failed").summary == counts(done=1) and ws.aside("t") == []


def test_a_done_file_an_interrupted_attempt_left_is_set_aside_too(ws):
    ws.write(ws.line("t", "donehold"))                                 # it writes its result, then keeps running
    with LivePool(ws, "--slots-per-gpu", "1", gpus="3") as pool:
        wait_until(lambda: ws.done("t").exists(), "the attempt to write its result")
        pool.proc.send_signal(signal.SIGTERM)
        assert pool.wait() == 128 + signal.SIGTERM
    assert not ws.done("t").exists() and [p.name for p in ws.aside("t")] == ["t.json.failed-a1"]
    assert ws.status() == [] and not ws.claim("t").exists()


def test_set_aside_never_overwrites_an_earlier_one_and_never_deletes(tmp_path):
    done = tmp_path / "t.json"
    done.write_text("first")
    first = taskpool.set_aside(str(done), 2)
    done.write_text("second")
    second = taskpool.set_aside(str(done), 2)
    assert (first, second) == (f"{done}.failed-a2", f"{done}.failed-a2.2")
    assert (Path(first).read_text(), Path(second).read_text()) == ("first", "second") and not done.exists()
    assert taskpool.set_aside(str(done), 2) is None                    # no file, nothing to set aside


# ------------------------------------------------------------------ checks before anything starts
@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
@pytest.mark.parametrize("where", ["log folder", "done folder", "status file"])
def test_a_folder_that_cannot_be_written_is_refused_before_anything_starts(ws, run, where):
    readonly = ws.root / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        bad_log, bad_done = ws.root / "logs" / "ok.log", ws.root / "out" / "ok.json"
        if where == "log folder":
            bad_log = readonly / "logs" / "ok.log"
        elif where == "done folder":
            bad_done = readonly / "out" / "ok.json"
        ws.write(f"ok\t{bad_done}\t{bad_log}\techo hi", ws.line("other"))
        extra = ["--status", str(readonly / "status.jsonl")] if where == "status file" else []
        for dry in ([], ["--dry-run"]):
            res = run(*extra, *dry)
            assert res.code == 2 and str(readonly) in res.err
        assert ws.ran("other") == [] and not (ws.root / "out").exists()
    finally:
        readonly.chmod(0o700)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_a_status_file_that_cannot_be_written_is_refused(ws, run):
    ws.write(ws.line("t"))
    ws.status_path.write_text("")
    ws.status_path.chmod(0o400)
    try:
        res = run()
        assert res.code == 2 and str(ws.status_path) in res.err and ws.ran("t") == []
    finally:
        ws.status_path.chmod(0o600)


# ------------------------------------------------------------------ the task file
BAD_LINES = {
    "too few fields": ("t2\t/abs/t2.json\t/abs/t2.log", "4 tab-separated fields"),
    "relative done_path": ("t2\tout/t2.json\t/abs/t2.log\techo hi", "done_path"),
    "relative log_path": ("t2\t/abs/t2.json\tlogs/t2.log\techo hi", "log_path"),
    "empty command": ("t2\t/abs/t2.json\t/abs/t2.log\t   ", "empty command"),
    "empty id": ("\t/abs/t2.json\t/abs/t2.log\techo hi", "empty id"),
    "duplicate id": ("t1\t/abs/t2.json\t/abs/t2.log\techo hi", "duplicate id 't1' (first on line 1)"),
    "duplicate done_path": ("t2\t{done_t1}\t/abs/t2.log\techo hi", "same done_path as line 1"),
    "log_path is another done_path": ("t2\t/abs/t2.json\t{done_t1}\techo hi", "log_path is the done_path of line 1"),
    "log_path is its own done_path": ("t2\t/abs/t2.json\t/abs/t2.json\techo hi",
                                      "log_path is the done_path of line 3"),
}


@pytest.mark.parametrize("case", list(BAD_LINES))
def test_a_bad_task_file_is_refused_naming_the_line_before_anything_runs(ws, run, case):
    bad, fragment = BAD_LINES[case]
    ws.write(ws.line("t1"), "# a comment", bad.format(done_t1=ws.done("t1")))
    res = run()
    assert res.code == 2
    assert "line 3" in res.err and fragment in res.err                # physical line number, comments included
    assert ws.ran("t1") == [] and not ws.claim("t1").exists() and not ws.log("t1").parent.exists()
    assert ws.status() == []


def test_every_bad_line_is_reported_at_once(ws, run):
    ws.write(ws.line("t1"), "t2\trelative\t/abs/t2.log\techo hi", ws.line("t3"), "t4\t/abs/t4.json")
    res = run()
    assert res.code == 2 and "line 2" in res.err and "line 4" in res.err and "line 3" not in res.err


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_blank_and_comment_lines_are_ignored_and_crlf_line_endings_tolerated(ws, run, newline):
    ws.write("", "# a comment", ws.line("t1"), "   ", "#another\tcomment\twith\ttabs", ws.line("t2"))
    ws.tsv.write_bytes(ws.tsv.read_bytes().replace(b"\n", newline))
    res = run()
    assert res.code == 0 and res.summary["ran_ok"] == 2


@pytest.mark.parametrize("content, message", [(None, "tasks.tsv"), ("# nothing here", "no tasks")])
def test_a_missing_or_empty_task_file_is_refused(ws, run, content, message):
    if content is not None:
        ws.write(content)
    res = run()
    assert res.code == 2 and message in res.err


def test_fields_are_stripped_so_padding_and_a_stray_carriage_return_do_no_harm():
    (task,) = taskpool.parse_tasks(" t \t /a/t.json \t /a/t.log \t echo hi \r\n")
    assert (task.id, task.done_path, task.log_path, task.command) == ("t", "/a/t.json", "/a/t.log", "echo hi")


def test_the_command_is_the_rest_of_the_line_tabs_included():
    (task,) = taskpool.parse_tasks("t\t/a/t.json\t/a/t.log\tprintf 'a\tb'\n")
    assert (task.id, task.done_path, task.log_path, task.command, task.line) == (
        "t", "/a/t.json", "/a/t.log", "printf 'a\tb'", 1)


@pytest.mark.parametrize("spec, expected", [
    ("0,1,2", ["0", "1", "2"]), (" 3 , 5 ", ["3", "5"]), ("0,0,1", ["0", "1"]),
    ("GPU-aa11,GPU-bb22", ["GPU-aa11", "GPU-bb22"]), ("", []), (None, []), (" , ,", [])])
def test_parse_gpus_keeps_distinct_ids_in_order(spec, expected):
    assert taskpool.parse_gpus(spec) == expected


def test_gpus_must_be_a_subset_of_cuda_visible_devices_when_both_are_set(ws, run, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    ws.write(ws.line("t"))
    res = run(gpus="2,5,6")
    assert res.code == 2 and "5, 6" in res.err and "CUDA_VISIBLE_DEVICES" in res.err and ws.ran("t") == []
    assert run("--dry-run", gpus="2,5,6").code == 2                    # a dry run is a faithful preview
    assert run(gpus="3,1").code == 0                                   # a subset passes, in any order
    assert {ws.result("t")["CUDA_VISIBLE_DEVICES"]} <= {"3", "1"}


@pytest.mark.parametrize("visible", ["", "  "])
def test_an_empty_cuda_visible_devices_does_not_limit_gpus(ws, run, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    ws.write(ws.line("t"))
    assert run(gpus="8,9").code == 0
