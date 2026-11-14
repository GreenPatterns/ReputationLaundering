"""GPU task pool (Task 4.1a): runs many single-GPU shell commands inside one Slurm allocation.

Phase 4 runs about a thousand trainings in a few allocations. The controller submits each allocation as
  sbatch --gres=gpu:a6000:N ... python -m keeper.cluster.taskpool TASKS.tsv --cwd <a fixed-commit worktree>
and several allocations may work through the same task list at the same time. Standard library only, POSIX only; the
pool itself never touches a GPU. Claims, status history and done files are in keeper.cluster.taskpool_state, the task
file and the start-up checks in keeper.cluster.taskpool_setup, the children in keeper.cluster.taskpool_process.

  python -m keeper.cluster.taskpool TASKS.tsv [--slots-per-gpu S] [--gpus 0,1,...] [--cwd DIR] [--status FILE]
         [--retries 1] [--poll 5] [--min-runtime 60] [--max-consecutive-failures 4] [--steal-stale]
         [--rerun-failed] [--dry-run]

Task file. One task per line, four tab-separated fields: id, done_path, log_path, command; a bad file ends the program
with status 2 and a message naming every bad line. A line, with <TAB> for a tab:
  e3_a_s5<TAB>/r/e3_a_s5.json<TAB>/r/logs/e3_a_s5.log<TAB>python -m keeper.cppr.cp_vpe_gpu ... --out /r/e3_a_s5.json

GPUs and slots. The GPUs are --gpus or, by default, the ids in CUDA_VISIBLE_DEVICES (Slurm sets it); with both, --gpus
must be a subset. The pool refuses to start without a GPU. Each GPU offers --slots-per-gpu slots (default 2), and a task
goes to the GPU running the fewest tasks. A task runs as `bash -c COMMAND` in --cwd (default: the current directory), in
its own process group, with the inherited environment plus CUDA_VISIBLE_DEVICES=<exactly one GPU id> and
CUBLAS_WORKSPACE_CONFIG=:4096:8. Its stdout and stderr are appended to log_path (folders created), after one header line
per attempt: task id, attempt, host, GPU, UTC time.

Success and retries. An attempt succeeds if the command exits 0 and done_path exists afterwards; exit 0 without
done_path is a failure. A failed attempt is repeated up to --retries times (default 1, the preregistered rule: a
crashed run is rerun once with the same seed), then the task is failed. The budget belongs to the task list, not to a
pool: before it starts a task, a pool reads the attempts that every pool recorded for it in the status file. A task
whose budget is spent is never started again, by this pool or by a later or concurrent one; it is reported as missing
(summary count `missing`, a status line of outcome `missing`). An interrupt uses up no attempt, so the next pool
takes over the rest of the budget. --rerun-failed ignores the history on purpose, starts at attempt 1 and marks its
status lines `rerun_failed`.

Done files. A task whose done_path exists is done and skipped, unless the status history shows a failed attempt behind
the file. A done file that an attempt left behind when it failed or was interrupted is renamed to
<done_path>.failed-a<attempt> (never deleted; the status line names it as `done_moved_to`), and so is one that a
recorded failure stands behind; the task then goes on as pending, or as missing if its budget is spent.

Faulty GPUs. A retry runs on a different GPU than the attempt that failed, and waits for one rather than reuse it; a
pool with no other usable GPU reuses it. A GPU is dropped after two failures in a row on it, or after a failure shorter
than --min-runtime seconds (default 60; 0 turns that off). After --max-consecutive-failures failures in a row on any
GPUs (default 4), or when no GPU is left, the pool stops cleanly: nothing new starts, running tasks finish, claims are
released, the rest of the task list stays untouched, and the exit status is 3. A task command that cannot even be
launched (a log folder that cannot be made) is a failed attempt, not a crash.

Claims and signals. A claimed task is skipped; this pool looks at it once and never takes it over later. On SIGTERM,
SIGINT or SIGHUP the pool first records the children that had already exited, then ends the running ones (SIGTERM to
each process group, SIGKILL after TERM_GRACE_SECONDS to every group that still has a member), then releases the
claims, and exits with 128 + the signal number. An interrupted attempt writes no status line, and no task starts once a
stop is requested.

Status. One JSON line per attempt is appended to --status (default TASKS.tsv.status.jsonl) under an exclusive flock:
id, attempt, host, pool, gpu, start_utc, end_utc, seconds, exit_code (negative: killed by that signal; null: not
launched) and outcome: ok, retry (failed, another attempt follows), failed (final) or missing (budget spent, nothing
ran). Optional keys: done_moved_to, error (the launch error), rerun_failed. The pools of a task list must share it.

Scheduling and exit. Free slots are filled in task-file order, retries first; the pool looks at its children every
WAKE_SECONDS and re-examines the schedule at least every --poll seconds. It ends when every task is done, failed,
missing or claimed by another pool, prints `summary: done=D ran_ok=R failed=F skipped_claimed=C missing=M` (D counts
tasks that were already done when the pool reached them) and exits 0 if nothing failed in this run (missing tasks are
reported; they failed in an earlier run), 1 if a task failed, 3 if the pool stopped, 128 + n after signal n. Status 2
means the pool refused to start. --dry-run validates the task file, the GPUs and the folders, prints the plan (counts,
slot count, pending tasks) and runs nothing: no claim, log or status file is created.
"""
import argparse
import collections
import dataclasses
import math
import os
import signal
import socket
import subprocess
import sys
import time

from keeper.cluster.taskpool_process import CUBLAS_WORKSPACE_CONFIG, handled_signals, launch, terminate_all
from keeper.cluster.taskpool_setup import (SetupError, Task, check_writable, load_tasks, parse_gpus, parse_tasks,
                                           resolve_cwd, resolve_gpus)
from keeper.cluster.taskpool_state import (append_status, budget_spent, current_attempts, is_stale, pid_alive,
                                           read_claim, read_history, release_claim, set_aside, set_claim_pgid,
                                           steal_stale, try_claim, utc_now)

__all__ = ["CUBLAS_WORKSPACE_CONFIG", "SetupError", "Task", "Settings", "GpuHealth", "Pool", "append_status",
           "budget_spent", "build_parser", "current_attempts", "is_stale", "launch", "load_tasks", "main",
           "parse_gpus", "parse_tasks", "pid_alive", "plan_text", "read_claim", "read_history", "release_claim",
           "set_aside", "set_claim_pgid", "steal_stale", "terminate_all", "try_claim"]

STATUS_SUFFIX = ".status.jsonl"
DEFAULT_SLOTS_PER_GPU = 2
DEFAULT_RETRIES = 1                      # the preregistered rule: a crashed run is rerun once
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_MIN_RUNTIME = 60.0               # a failure faster than this drops the GPU it ran on
DEFAULT_MAX_CONSECUTIVE_FAILURES = 4     # failures in a row, on any GPUs, that stop the pool
GPU_STRIKES = 2                          # failures in a row on one GPU that drop it
WAKE_SECONDS = 0.1                       # how often the wait between polls looks at the children and at signals
EXIT_REFUSED = 2                         # bad task file, no GPU, bad --cwd, unwritable folder
EXIT_STOPPED = 3                         # too many failures in a row, or no usable GPU left


@dataclasses.dataclass(frozen=True)
class Settings:
    tasks: list
    gpus: tuple
    slots_per_gpu: int
    cwd: str
    status_path: str
    retries: int
    poll: float
    steal_stale: bool
    rerun_failed: bool = False
    min_runtime: float = DEFAULT_MIN_RUNTIME
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES


@dataclasses.dataclass(frozen=True)
class Running:
    task: Task
    attempt: int
    gpu: str
    proc: subprocess.Popen | None        # None: the attempt could not be launched
    start_utc: str
    started: float                       # time.monotonic() when the child was launched


def say(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def pool_id(host: str) -> str:
    """Names this pool in its claims and status lines: the Slurm job (if any), the host and the pid."""
    job = os.environ.get("SLURM_JOB_ID")
    return f"{host}:{os.getpid()}" if not job else f"slurm{job}:{host}:{os.getpid()}"


# ------------------------------------------------------------------ GPU health
class GpuHealth:
    """Which GPUs may still take tasks, and whether the pool should stop.

    A GPU is dropped after GPU_STRIKES failures in a row on it, or after a failure shorter than min_runtime seconds (a
    task that dies at once points at the GPU or the node more than at the task); a success on a GPU clears its strikes
    but does not bring a dropped GPU back. After max_consecutive failures in a row on any GPUs, `halted` says why the
    pool should stop. An attempt that never started (gpu None) counts in the run of failures but not against a GPU."""

    def __init__(self, gpus, min_runtime: float, max_consecutive: int):
        self.gpus = tuple(gpus)
        self.min_runtime, self.max_consecutive = min_runtime, max_consecutive
        self.strikes = collections.Counter()
        self.dropped = {}                # GPU -> why
        self.streak = 0                  # failures in a row, on any GPUs
        self.halted = None               # why the pool should stop, once it should

    def usable(self) -> list:
        return [gpu for gpu in self.gpus if gpu not in self.dropped]

    def note(self, gpu, ok: bool, seconds: float) -> list:
        """Account for an attempt that ended on gpu after `seconds`; returns what to tell the operator."""
        if ok:
            self.streak = 0
            self.strikes[gpu] = 0
            return []
        messages = []
        self.streak += 1
        if gpu is not None and gpu not in self.dropped:
            self.strikes[gpu] += 1
            why = self._why_drop(gpu, seconds)
            if why:
                self.dropped[gpu] = why
                messages.append(f"dropping GPU {gpu}: {why}")
        if self.streak >= self.max_consecutive and self.halted is None:
            self.halted = f"{self.streak} failures in a row"
            messages.append(f"stopping after {self.streak} failures in a row (--max-consecutive-failures)")
        return messages

    def _why_drop(self, gpu, seconds: float) -> str:
        if seconds < self.min_runtime:
            return f"a failure after {seconds:.0f} s, shorter than --min-runtime {self.min_runtime:g} s"
        if self.strikes[gpu] >= GPU_STRIKES:
            return f"{self.strikes[gpu]} failures in a row on it"
        return ""


# ------------------------------------------------------------------ the pool
class Pool:
    """Works through the tasks of a Settings on the GPUs' slots; run() returns the exit status."""

    def __init__(self, settings: Settings):
        self.cfg = settings
        self.host = socket.gethostname()
        self.pool_id = pool_id(self.host)
        self.waiting = collections.deque(settings.tasks)    # not looked at yet, in task-file order
        self.retry_queue = []                               # (task, next attempt, GPU that failed); the claim is held
        self.running = []
        self.held = {}                                      # task id -> task, for every claim this pool holds
        self.health = GpuHealth(settings.gpus, settings.min_runtime, settings.max_consecutive_failures)
        self.n_done = self.n_ok = 0
        self.failed, self.missing, self.skipped = [], [], []    # task ids
        self._stop = None                                   # the number of the first stop signal received

    def run(self) -> int:
        """Work until every task is done, failed, missing or claimed elsewhere, or until a stop. 0 if nothing failed in
        this run, 1 if a task failed, 3 if the pool stopped itself, 128 + n if signal n ended the run."""
        os.makedirs(os.path.dirname(os.path.abspath(self.cfg.status_path)), exist_ok=True)
        say(f"pool {self.pool_id}: GPUs {','.join(self.cfg.gpus)} x {self.cfg.slots_per_gpu} slots, "
            f"{len(self.cfg.tasks)} tasks" + (" (--rerun-failed: recorded failures are ignored)"
                                              if self.cfg.rerun_failed else ""))
        with handled_signals(self.request_stop):
            try:
                self._loop()
            finally:
                self._shutdown()
        return self.exit_code()

    def request_stop(self, signum, frame=None) -> None:
        """The signal handler: remember the first signal; the loop sees it within WAKE_SECONDS."""
        if self._stop is None:
            self._stop = int(signum)

    def exit_code(self) -> int:
        if self._stop:
            return 128 + self._stop
        if self.health.halted:
            return EXIT_STOPPED
        return int(bool(self.failed))

    def summary(self) -> str:
        lines = []
        if self._stop:
            lines.append(f"interrupted by {signal.Signals(self._stop).name}: unfinished tasks are free again")
        if self.health.halted:
            lines.append(f"stopped: {self.health.halted}; {len(self.waiting)} task(s) not started, "
                         f"{len(self.retry_queue)} retry(ies) left for the next pool")
        lines.append(f"summary: done={self.n_done} ran_ok={self.n_ok} failed={len(self.failed)} "
                     f"skipped_claimed={len(self.skipped)} missing={len(self.missing)}")
        for title, ids in (("failed tasks", self.failed), ("missing tasks", self.missing),
                           ("claimed by another pool", self.skipped)):
            if ids:
                lines.append(f"{title}: " + " ".join(ids))
        return "\n".join(lines)

    # -- scheduling
    def _stopping(self) -> bool:
        return self._stop is not None or self.health.halted is not None

    def _loop(self) -> None:
        while self._stop is None:
            self._reap()
            self._fill_slots()
            if not self.running:                            # nothing runs and nothing could start
                if (self.waiting or self.retry_queue) and self.health.halted is None:
                    self.health.halted = "no usable GPU left"
                return
            self._sleep()

    def _sleep(self) -> None:
        """Wait up to --poll seconds, but return at once for a signal or a child that has exited."""
        deadline = time.monotonic() + self.cfg.poll
        while self._stop is None and time.monotonic() < deadline:
            if any(run.proc.poll() is not None for run in self.running):
                return
            time.sleep(min(WAKE_SECONDS, max(0.0, deadline - time.monotonic())))

    def _free_gpu(self, candidates=None) -> str | None:
        """The usable GPU (among candidates, if given) with a free slot that runs the fewest tasks, or None."""
        load = collections.Counter(run.gpu for run in self.running)
        pool = self.health.usable() if candidates is None else candidates
        free = [gpu for gpu in pool if load[gpu] < self.cfg.slots_per_gpu]
        return min(free, key=load.__getitem__) if free else None

    def _gpu_for_retry(self, failed_on: str) -> str | None:
        """A GPU for a retry: a different one than the failed attempt's, the same one only if no other is usable."""
        usable = self.health.usable()
        others = [gpu for gpu in usable if gpu != failed_on]
        return self._free_gpu(others if others else usable)

    def _fill_slots(self) -> None:
        """Start what can start, the retries first, until a step places nothing; nothing starts once a stop is due."""
        while not self._stopping() and (self._place_retry() or self._place_next()):
            pass

    def _place_retry(self) -> bool:
        for index, (task, attempt, failed_on) in enumerate(self.retry_queue):
            gpu = self._gpu_for_retry(failed_on)
            if gpu is not None:
                del self.retry_queue[index]
                self._start(task, attempt, gpu)
                return True
        return False

    def _place_next(self) -> bool:
        gpu = self._free_gpu()
        if gpu is None or not self.waiting:
            return False
        task = self.waiting.popleft()
        verdict = self._begin(task, gpu)
        if verdict == "done":
            self.n_done += 1
        elif verdict == "taken":
            self.skipped.append(task.id)
        return True

    # -- one task
    def _attempts(self, task: Task) -> list:
        """The recorded attempts that count against the task's budget (none with --rerun-failed)."""
        if self.cfg.rerun_failed:
            return []
        return current_attempts(read_history(self.cfg.status_path, task.id))

    def _begin(self, task: Task, gpu: str) -> str:
        """Start the task if it is free: 'done', 'taken' (another pool owns it), 'missing' (budget spent), 'started'."""
        if os.path.exists(task.done_path) and not self._attempts(task):
            return "done"                                   # a done file counts unless a failed attempt is behind it
        if not self._claim(task):
            return "taken"
        spent = self._attempts(task)                        # read again under the claim: other pools record meanwhile
        if os.path.exists(task.done_path):
            if not spent:
                self._release(task)
                return "done"                               # finished by another pool between the check and the claim
            self._set_aside(task, spent[-1]["attempt"], "a failed attempt is on record for it")
        if budget_spent(spent, self.cfg.retries):
            self._record_missing(task, spent)
            self._release(task)
            return "missing"
        self._start(task, len(spent) + 1, gpu)
        return "started"

    def _claim(self, task: Task) -> bool:
        """Take the task's claim, stealing a stale one with --steal-stale. True if this pool now holds it."""
        owner = dict(host=self.host, pid=os.getpid(), pgid=None, pool=self.pool_id, utc=utc_now())
        won = try_claim(task.claim_path, owner)
        if not won and self.cfg.steal_stale and steal_stale(task.claim_path, self.host):
            won = try_claim(task.claim_path, owner)
        if won:
            self.held[task.id] = task
        return won

    def _release(self, task: Task) -> None:
        release_claim(task.claim_path, self.pool_id)
        self.held.pop(task.id, None)

    def _set_aside(self, task: Task, attempt: int, why: str) -> None:
        try:
            moved = set_aside(task.done_path, attempt)
        except OSError as error:
            say(f"could not set aside the done file of {task.id}: {error}")
            return
        if moved:
            say(f"set aside {moved}: {why}")

    def _start(self, task: Task, attempt: int, gpu: str) -> None:
        started = utc_now()
        header = f"[taskpool] task={task.id} attempt={attempt} host={self.host} gpu={gpu} utc={started}\n"
        try:
            proc = launch(task, gpu, self.cfg.cwd, header)
        except OSError as error:                            # this task's problem (a log folder), not the pool's
            say(f"launch of {task.id} failed: {error}")
            self._finish(Running(task, attempt, gpu, None, started, time.monotonic()), None, str(error))
            return
        self.running.append(Running(task, attempt, gpu, proc, started, time.monotonic()))
        set_claim_pgid(task.claim_path, self.pool_id, proc.pid)
        say(f"start {task.id} attempt {attempt} gpu {gpu}")

    def _reap(self) -> None:
        """Account for every child that has exited."""
        still_running = []
        for run in self.running:
            code = run.proc.poll()
            if code is None:
                still_running.append(run)
            else:
                self._finish(run, code)
        self.running = still_running

    def _finish(self, run: Running, code: int | None, error: str | None = None) -> None:
        """Account for an ended attempt: set aside the done file it left, record it, then retry, settle or release."""
        ok = code == 0 and os.path.exists(run.task.done_path)
        moved = None if ok else set_aside(run.task.done_path, run.attempt)
        outcome = "ok" if ok else ("retry" if run.attempt <= self.cfg.retries else "failed")
        seconds = round(time.monotonic() - run.started, 3)
        self._record(run, code, outcome, seconds, moved, error)
        for message in self.health.note(None if error else run.gpu, ok, seconds):
            say(message)
        if outcome == "retry":
            self.retry_queue.append((run.task, run.attempt + 1, run.gpu))
            return
        if ok:
            self.n_ok += 1
        else:
            self.failed.append(run.task.id)
        self._release(run.task)

    def _status_line(self, task: Task, attempt: int, gpu, start_utc: str, end_utc: str, seconds: float, exit_code,
                     outcome: str, **optional) -> dict:
        line = dict(id=task.id, attempt=attempt, host=self.host, pool=self.pool_id, gpu=gpu, start_utc=start_utc,
                    end_utc=end_utc, seconds=seconds, exit_code=exit_code, outcome=outcome)
        if self.cfg.rerun_failed:
            line["rerun_failed"] = True
        return {**line, **{key: value for key, value in optional.items() if value is not None}}

    def _record(self, run: Running, code, outcome: str, seconds: float, moved, error) -> None:
        append_status(self.cfg.status_path, self._status_line(run.task, run.attempt, run.gpu, run.start_utc, utc_now(),
                                                              seconds, code, outcome, done_moved_to=moved, error=error))
        notes = ([f"launch failed: {error}"] if error else []) + (
            ["exit 0 but no done file"] if code == 0 and outcome != "ok" else []) + (
            [f"done file set aside as {moved}"] if moved else [])
        say(f"{outcome} {run.task.id} attempt {run.attempt} gpu {run.gpu} exit {code} after {seconds:.1f}s"
            + (f" ({'; '.join(notes)})" if notes else ""))

    def _record_missing(self, task: Task, spent: list) -> None:
        now = utc_now()
        append_status(self.cfg.status_path, self._status_line(task, len(spent), None, now, now, 0.0, None, "missing"))
        self.missing.append(task.id)
        say(f"missing {task.id}: its retry budget is spent ({len(spent)} attempt(s) on record); "
            "--rerun-failed runs it again")

    def _shutdown(self) -> None:
        """Account for children that already ended, end the rest and set aside what they left, release every claim."""
        try:
            self._reap()
        finally:
            interrupted = [run for run in self.running if run.proc.poll() is None]
            if interrupted:
                terminate_all([run.proc for run in interrupted])
                for run in interrupted:
                    self._set_aside(run.task, run.attempt, f"attempt {run.attempt} of {run.task.id} was interrupted")
                self.running = []
            released = list(self.held.values())
            for task in released:
                self._release(task)
            if interrupted or released:
                say(f"stopped early: ended running {' '.join(run.task.id for run in interrupted) or 'none'}; "
                    f"released {len(released)} claim(s)")


# ------------------------------------------------------------------ plan and command line
def _plan_status(task: Task, settings: Settings, host: str) -> str:
    """'done', 'claimed' (another pool owns it), 'missing' (budget spent) or 'pending', as the pool would see it."""
    attempts = [] if settings.rerun_failed else current_attempts(read_history(settings.status_path, task.id))
    if os.path.exists(task.done_path) and not attempts:
        return "done"
    if os.path.exists(task.claim_path):
        claim = read_claim(task.claim_path)
        if not (settings.steal_stale and claim is not None and is_stale(claim, host)):
            return "claimed"
    return "missing" if budget_spent(attempts, settings.retries) else "pending"


def plan_text(settings: Settings, host: str) -> str:
    """The --dry-run plan: GPUs and slots, the task counts and the pending tasks in task-file order."""
    status = {task.id: _plan_status(task, settings, host) for task in settings.tasks}
    count = collections.Counter(status.values())
    pending = [task for task in settings.tasks if status[task.id] == "pending"]
    lines = ["dry run: nothing is started",
             f"GPUs {','.join(settings.gpus)} x {settings.slots_per_gpu} slots per GPU = "
             f"{len(settings.gpus) * settings.slots_per_gpu} slots",
             f"{len(settings.tasks)} tasks: {count['done']} done, {count['claimed']} claimed by another pool, "
             f"{count['missing']} missing, {count['pending']} pending"]
    if pending:
        lines.append("pending, in task-file order:")
        lines += [f"  {task.id}  {task.done_path}" for task in pending]
    return "\n".join(lines)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} is negative")
    return value


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"{text} is not a positive number of seconds")
    return value


def _non_negative_seconds(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value >= 0):
        raise argparse.ArgumentTypeError(f"{text} is not a number of seconds that is 0 or more")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m keeper.cluster.taskpool",
                                 description="Run single-GPU shell commands from a task file on the GPUs of one "
                                             "allocation, with claims, a retry budget per task list and a status log.")
    ap.add_argument("tasks", metavar="TASKS.tsv", help="one task per line: id, done_path, log_path, command "
                                                       "(tab-separated)")
    ap.add_argument("--slots-per-gpu", type=_positive_int, default=DEFAULT_SLOTS_PER_GPU, metavar="S",
                    help="concurrent tasks per GPU (default %(default)s)")
    ap.add_argument("--gpus", default=None, metavar="IDS",
                    help="comma-separated GPU ids (default: the ids in CUDA_VISIBLE_DEVICES); with both set, a "
                         "subset of it")
    ap.add_argument("--cwd", default=os.curdir, metavar="DIR",
                    help="working directory of every task (default: the current directory)")
    ap.add_argument("--status", default=None, metavar="FILE",
                    help="status log and retry budget, one JSON line per attempt (default: TASKS.tsv.status.jsonl); "
                         "the pools of one task list must share it")
    ap.add_argument("--retries", type=_non_negative_int, default=DEFAULT_RETRIES,
                    help="reruns of a failed task (default %(default)s, the preregistered rule)")
    ap.add_argument("--poll", type=_positive_seconds, default=DEFAULT_POLL_SECONDS, metavar="SECONDS",
                    help="longest wait between schedule checks (default %(default)s)")
    ap.add_argument("--min-runtime", type=_non_negative_seconds, default=DEFAULT_MIN_RUNTIME, metavar="SECONDS",
                    help="a failure faster than this drops the GPU it ran on (default %(default)s; 0 turns it off)")
    ap.add_argument("--max-consecutive-failures", type=_positive_int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
                    metavar="N", help="failures in a row, on any GPUs, that stop the pool (default %(default)s)")
    ap.add_argument("--steal-stale", action="store_true",
                    help="release claims of dead pools on this host whose attempt's process group is gone")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="ignore the recorded history: run tasks whose retry budget is spent again, from attempt 1")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings(tasks=load_tasks(args.tasks), gpus=resolve_gpus(args.gpus),
                            slots_per_gpu=args.slots_per_gpu, cwd=resolve_cwd(args.cwd),
                            status_path=os.path.abspath(args.status or args.tasks + STATUS_SUFFIX),
                            retries=args.retries, poll=args.poll, steal_stale=args.steal_stale,
                            rerun_failed=args.rerun_failed, min_runtime=args.min_runtime,
                            max_consecutive_failures=args.max_consecutive_failures)
        check_writable(settings.tasks, settings.status_path)
    except SetupError as error:
        print(f"taskpool: {error}", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        print(plan_text(settings, socket.gethostname()))
        return 0
    pool = Pool(settings)
    code = pool.run()
    print(pool.summary(), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
