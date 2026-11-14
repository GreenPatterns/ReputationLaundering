"""Helpers shared by test_taskpool.py, test_taskpool_state.py and test_taskpool_health.py (not collected).

CHILD is the dummy task: a small Python script that writes its done file, the environment it saw and how often it ran
(Workspace.ran). Nothing touches a GPU or the stored results (ruling R27): every file lives in pytest's tmp_path.
The test modules take the fixtures ws, run and dead_pid from here (`ws, run = common.ws, common.run`), so that the
fixture names stay usable as test arguments. Most tests call taskpool.main in this process; the tests that send the
pool a signal or run several pools at once start it as `python -m keeper.cluster.taskpool` (LivePool).

Timing: tests never assert how long something took. They wait on files with generous limits, and the concurrency test
uses a barrier (the first wave of children waits until all of it has started), so that reaching the full slot count
is forced and exceeding it shows in the children's own start and end times.
"""
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import pytest

from keeper.cluster import taskpool

ROOT = Path(__file__).resolve().parents[1]
UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
SUMMARY = re.compile(r"summary: done=(\d+) ran_ok=(\d+) failed=(\d+) skipped_claimed=(\d+) missing=(\d+)")
SUMMARY_FIELDS = ("done", "ran_ok", "failed", "skipped_claimed", "missing")
DEAD_PID = 2 ** 22 + 1          # above any pid the kernel hands out (pid_max <= 2**22): dead now and for ever
OWNER = dict(host=socket.gethostname(), pid=os.getpid(), pgid=None, pool="another-pool", utc="2026-10-27T00:00:00Z")
FAST = ["--poll", "0.05", "--min-runtime", "0", "--max-consecutive-failures", "1000"]   # no GPU drops, no halt

CHILD = r'''
import json, os, sys, time
from pathlib import Path

mode, done, arg = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
task, claim, folder = done.stem, Path(str(done) + ".claim"), done.parent
folder.mkdir(parents=True, exist_ok=True)


def note(name, text):
    with open(folder / f"{task}.{name}", "a") as f:
        f.write(text + "\n")


def write_done():
    seen = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "KEEPER_TEST_MARK")}
    done.write_text(json.dumps(dict(seen, cwd=os.getcwd(), pid=os.getpid())))


def behave(what):
    if what == "ok":
        write_done()
    elif what == "fail":
        print("boom on stdout")
        print("boom on stderr", file=sys.stderr)
        sys.exit(3)
    elif what == "nodone":
        pass
    elif what == "donefail":
        write_done()
        sys.exit(1)
    elif what == "donehold":
        write_done()
        time.sleep(120)
    elif what == "hold":
        time.sleep(120)                 # the cap keeps a failed test from leaving the process behind


note("ran", f"{os.getpid()} claim={claim.exists()}")
runs = len((folder / f"{task}.ran").read_text().splitlines())          # this run's number: 1 for the first
if mode == "seq":                       # run n behaves as the n-th word of arg[0]; the last word repeats
    words = arg[0].split(",")
    behave(words[min(runs, len(words)) - 1])
elif mode == "sleep":                   # takes arg[0] seconds, then succeeds
    time.sleep(float(arg[0]))
    write_done()
elif mode == "gate":                    # succeeds once the file arg[0] exists: a test holds the task till then
    deadline = time.time() + 60
    while not Path(arg[0]).exists() and time.time() < deadline:
        time.sleep(0.02)
    write_done()
elif mode == "failon":                  # fails on the GPU named arg[0] and succeeds on every other
    behave("fail" if os.environ["CUDA_VISIBLE_DEVICES"] == arg[0] else "ok")
elif mode == "barrier":
    record, n_start, hold = Path(arg[0]), int(arg[1]), float(arg[2])
    record.mkdir(parents=True, exist_ok=True)
    start = time.time()
    (record / f"start_{task}").write_text("")
    deadline = time.time() + 30
    while len(list(record.glob("start_*"))) < n_start:
        if time.time() > deadline:
            sys.exit(5)
        time.sleep(0.01)
    time.sleep(hold)
    span = dict(start=start, end=time.time(), gpu=os.environ["CUDA_VISIBLE_DEVICES"])
    (record / f"span_{task}.json").write_text(json.dumps(span))
    write_done()
else:
    behave(mode)
'''


class Outcome(NamedTuple):
    code: int
    out: str
    err: str

    @property
    def summary(self) -> dict:
        return parse_summary(self.out)


def parse_summary(text: str) -> dict:
    match = SUMMARY.search(text)
    assert match, text
    return dict(zip(SUMMARY_FIELDS, map(int, match.groups())))


def counts(**given) -> dict:
    """The summary counts with 0 for every one not given."""
    return {key: given.get(key, 0) for key in SUMMARY_FIELDS}


class Workspace:
    """One test's files under tmp_path: out/<id>.json done files, logs/<id>.log, tasks.tsv and the child script."""

    def __init__(self, root: Path):
        self.root, self.tsv, self.child = root, root / "tasks.tsv", root / "child.py"
        self.child.write_text(CHILD)

    def done(self, task_id) -> Path:
        return self.root / "out" / f"{task_id}.json"

    def claim(self, task_id) -> Path:
        return self.root / "out" / f"{task_id}.json.claim"

    def log(self, task_id) -> Path:
        return self.root / "logs" / f"{task_id}.log"

    def aside(self, task_id) -> list:
        """The done files a failed or interrupted attempt left, renamed to <done>.failed-a<attempt>."""
        return sorted((self.root / "out").glob(f"{task_id}.json.failed-a*"))

    def line(self, task_id, mode="ok", *args, command=None) -> str:
        """The TSV line of a task running CHILD in `mode` (or the given shell command)."""
        command = command or shlex.join([sys.executable, str(self.child), mode, str(self.done(task_id)), *args])
        return "\t".join([task_id, str(self.done(task_id)), str(self.log(task_id)), command])

    def write(self, *lines) -> None:
        self.tsv.write_text("\n".join(lines) + "\n")

    def put_claim(self, task_id, owner) -> None:
        self.claim(task_id).parent.mkdir(parents=True, exist_ok=True)
        self.claim(task_id).write_text(owner if isinstance(owner, str) else json.dumps(owner))

    def put_status(self, *rows, path=None) -> None:
        """Append rows (dicts) to the status file, as pools that ran earlier would have."""
        with open(path or self.status_path, "a") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)

    @property
    def status_path(self) -> Path:
        return Path(str(self.tsv) + ".status.jsonl")

    def ran(self, task_id) -> list:
        """One entry per run of the task's command: '<pid> claim=<whether its claim file existed>'."""
        path = self.root / "out" / f"{task_id}.ran"
        return path.read_text().split("\n")[:-1] if path.exists() else []

    def child_pid(self, task_id) -> int:
        return int(self.ran(task_id)[0].split()[0])

    def result(self, task_id) -> dict:
        return json.loads(self.done(task_id).read_text())

    def status(self, path=None) -> list:
        path = Path(path or self.status_path)
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def attempts(self, task_id) -> list:
        """(attempt, outcome) of the task's status lines, in file order."""
        return [(row["attempt"], row["outcome"]) for row in self.status() if row["id"] == task_id]

    def settings(self, gpus=("3", "5"), slots=1, **overrides) -> "taskpool.Settings":
        """Settings of the task list for a Pool built in the test (default status path, no GPU drops, no halt)."""
        fields = dict(tasks=taskpool.load_tasks(str(self.tsv)), gpus=tuple(gpus), slots_per_gpu=slots,
                      cwd=str(self.root), status_path=str(self.status_path), retries=1, poll=0.05, steal_stale=False,
                      rerun_failed=False, min_runtime=0.0, max_consecutive_failures=1000)
        return taskpool.Settings(**{**fields, **overrides})


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)     # the tests name their GPUs, not the allocation's
    return Workspace(tmp_path)


@pytest.fixture
def run(ws, capsys):
    """run(*extra_args, gpus='3,5', cwd=True): taskpool.main on ws's task list in this process (FAST settings)."""
    def _run(*args, gpus="3,5", cwd=True):
        argv = [str(ws.tsv), *FAST] + (["--gpus", gpus] if gpus is not None else [])
        argv += ["--cwd", str(ws.root)] if cwd else []
        code = taskpool.main(argv + list(args))
        captured = capsys.readouterr()
        return Outcome(code, captured.out, captured.err)
    return _run


@pytest.fixture
def dead_pid() -> int:
    """A pid that no process has or can get, so that pid reuse cannot make it alive."""
    with pytest.raises(ProcessLookupError):
        os.kill(DEAD_PID, 0)
    return DEAD_PID


def wait_until(predicate, what, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def is_running(pid: int) -> bool:
    """True if the process exists and is not a zombie."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class LivePool:
    """The pool as a child process of the test: `python -m keeper.cluster.taskpool` from the repository root."""

    def __init__(self, ws: Workspace, *args, gpus="3,5", name="pool"):
        self.out_path = ws.root / f"{name}.out"
        argv = [sys.executable, "-m", "keeper.cluster.taskpool", str(ws.tsv), *FAST, "--gpus", gpus,
                "--cwd", str(ws.root), *args]
        with open(self.out_path, "w") as sink:                  # the child keeps its own copy of the descriptor
            self.proc = subprocess.Popen(argv, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)

    def wait(self, timeout=60) -> int:
        return self.proc.wait(timeout=timeout)

    @property
    def text(self) -> str:
        return self.out_path.read_text()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def max_overlap(spans: list) -> int:
    """The largest number of (start, end) spans that were running at the same instant."""
    events = sorted([(s["start"], 1) for s in spans] + [(s["end"], -1) for s in spans])
    running = peak = 0
    for _, step in events:
        running += step
        peak = max(peak, running)
    return peak
