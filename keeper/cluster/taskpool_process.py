"""The children of the task pool (Task 4.1a): starting a task's command, ending it, and routing the stop signals.

A task runs as `bash -c COMMAND` in its own process group (it leads a new session), so that the pool can end the whole
group: a command such as `cd x && python ...` leaves the python process as a grandchild of the shell, and a training
must not outlive its claim. terminate_all sends SIGTERM to every group, waits for the groups to empty (a leader that
obeyed does not mean that its members did), then sends SIGKILL to every group that still has a member. A group that
has no member cannot be signalled, and a process group id is not reused while it has a member, so a signal never
reaches an unrelated process.
"""
import contextlib
import os
import signal
import subprocess
import threading
import time

from keeper.cluster.taskpool_setup import Task
from keeper.cluster.taskpool_state import group_alive

CUBLAS_WORKSPACE_CONFIG = ":4096:8"      # deterministic cuBLAS, set for every task (global constraint)
TERM_GRACE_SECONDS = 10.0                # SIGTERM to SIGKILL for a task that ignores SIGTERM (Slurm's KillWait is 30 s)
KILL_WAIT_SECONDS = 5.0                  # how long the SIGKILLed groups get to disappear
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def launch(task: Task, gpu: str, cwd: str, header: str) -> subprocess.Popen:
    """Start the task's command through `bash -c` on one GPU, in its own process group.

    The header line is written to the log first; the command's stdout and stderr then append to the same file."""
    os.makedirs(os.path.dirname(task.log_path), exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=CUBLAS_WORKSPACE_CONFIG)
    with open(task.log_path, "ab") as log:
        log.write(header.encode("utf-8"))
        log.flush()
        return subprocess.Popen(["bash", "-c", task.command], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def _signal_group(pgid: int, signum: int) -> None:
    """Send signum to the process group pgid; a group without members cannot be signalled, and its id may be reused."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signum)


def _groups_alive(procs: list) -> bool:
    """True while any child's process group has a member; a leader that has ended is reaped on the way."""
    return any([proc.poll() is None or group_alive(proc.pid) for proc in procs])


def _wait_until_gone(procs: list, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and _groups_alive(procs):
        time.sleep(0.02)


def terminate_all(procs: list, grace=None) -> None:
    """End the process groups the children lead: SIGTERM to all, SIGKILL after `grace` seconds to every group that still
    has a member (also one whose leader obeyed and is gone), then reap the leaders."""
    grace = TERM_GRACE_SECONDS if grace is None else grace
    for proc in procs:
        _signal_group(proc.pid, signal.SIGTERM)
    _wait_until_gone(procs, grace)
    for proc in procs:
        _signal_group(proc.pid, signal.SIGKILL)
    _wait_until_gone(procs, KILL_WAIT_SECONDS)
    for proc in procs:
        proc.wait()


@contextlib.contextmanager
def handled_signals(handler):
    """Route the stop signals to handler inside the block (only the main thread can), then restore them."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, handler) for sig in STOP_SIGNALS}
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, signal.SIG_DFL if old is None else old)
