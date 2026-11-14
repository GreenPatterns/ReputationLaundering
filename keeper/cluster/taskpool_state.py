"""The shared state of one task list (Task 4.1a): the claims beside the done files and the status file's history.

Claims. A pool claims a task by creating done_path + ".claim" exclusively (O_CREAT | O_EXCL), holding the JSON
{host, pid, pgid, pool, utc}: the pool's host and pid, the process group of the attempt that runs now (null until
it starts) and the pool's name. A pool that finds the claim leaves the task alone. The claim is removed by its owner
after success and after final failure; a pool that was killed hard leaves its claim, and --steal-stale may take it
over when its host is this host, its pid no longer exists and no process is left in its attempt's process group
(an attempt can outlive its pool; running the task again beside it would be a duplicate). Same-host steals queue on
an flock of the claim file itself, and the claim is removed only if it still reads as the one judged stale.

Status history. The status file holds one JSON line per attempt of every pool, appended under an exclusive flock.
It is also the retry budget of the task list: a pool that has claimed a task reads the attempts recorded for it
under a shared flock. The attempts that count are those since the task's last success and since its last first
attempt (a deliberate rerun starts at attempt 1 again). The budget is spent when the task has a `failed` line, or
when its `retry` and `failed` lines reach retries + 1. Otherwise the next attempt number is the count plus one, so a
budget that an interrupt cut short carries over to whichever pool takes the task next. `missing` lines record that a
pool skipped a task with a spent budget; they are no attempts and the history ignores them.

Done files. An attempt that did not succeed may leave a done file (a driver writes its result before its dumps, a
kill can cut the write short). set_aside renames it to <done_path>.failed-a<attempt>, never deleting it, so that no
pool counts it as a result.
"""
import contextlib
import datetime
import fcntl
import json
import os

CLAIM_SUFFIX = ".claim"
ATTEMPT_OUTCOMES = ("ok", "retry", "failed")         # `missing` is the outcome of no attempt


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ claims
def try_claim(path: str, owner: dict) -> bool:
    """Create the claim file exclusively (O_CREAT | O_EXCL) holding owner as JSON; False if it already exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(owner, f)
    except BaseException:                # e.g. a full disk: an empty claim would block the task for good
        os.unlink(path)
        raise
    return True


def read_claim(path: str) -> dict | None:
    """The claim file's JSON object, or None if it is missing, unreadable or not an object."""
    try:
        with open(path) as f:
            claim = json.load(f)
    except (OSError, ValueError):
        return None
    return claim if isinstance(claim, dict) else None


def set_claim_pgid(path: str, pool: str, pgid: int) -> None:
    """Record the process group of the running attempt in the claim, if the claim is still this pool's."""
    claim = read_claim(path)
    if claim is None or claim.get("pool") != pool:
        return
    replacement = f"{path}.tmp.{os.getpid()}"
    try:
        with open(replacement, "w") as f:
            json.dump(dict(claim, pgid=pgid), f)
        os.replace(replacement, path)    # readers see the old claim or the new one, never half of one
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(replacement)
        raise


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return False                     # no /proc here: take the process as it looks to kill(pid, 0)


def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists and is not a zombie (a dead process nobody has reaped)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                      # it exists but belongs to another user
    return not _is_zombie(pid)


def group_alive(pgid: int) -> bool:
    """True if some process is in the process group pgid. A group without members has no id that is kept from reuse."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _valid_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_stale(claim: dict, host: str) -> bool:
    """True if the claim's owner ran on `host`, its pid no longer exists and its attempt's process group is empty.

    Anything unclear (another host, a pid or pgid that is not a number, a process that is still there) is not stale."""
    pid, pgid = claim.get("pid"), claim.get("pgid")
    return (claim.get("host") == host and _valid_id(pid) and not pid_alive(pid)
            and not (_valid_id(pgid) and group_alive(pgid)))


def steal_stale(path: str, host: str) -> bool:
    """Remove the claim at path if it is stale; True if the way is free for a new claim.

    Thieves of one claim all run on this host and queue on an flock of the claim file. Once a thief holds the lock it
    reads the claim again and removes it only if it still reads as the one that was judged stale; a pool that took the
    task over meanwhile keeps its claim."""
    judged = read_claim(path)
    if judged is None:
        return not os.path.lexists(path)             # gone: the way is free; unreadable: left alone
    if not is_stale(judged, host):
        return False
    try:
        locked = open(path, "r+")                    # NFS takes an exclusive flock only on a file open for writing
    except FileNotFoundError:
        return True
    except OSError:
        return False
    with locked:
        fcntl.flock(locked, fcntl.LOCK_EX)
        current = read_claim(path)
        if current is None:
            return not os.path.lexists(path)         # removed while this thief waited: free; unreadable: left alone
        if current != judged:
            return False                             # another pool took the task over meanwhile
        os.unlink(path)
        return True


def release_claim(path: str, pool: str) -> None:
    """Remove the claim at path if this pool owns it; another pool's claim is left alone."""
    claim = read_claim(path)
    if claim is not None and claim.get("pool") == pool:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


# ------------------------------------------------------------------ status file and retry budget
def append_status(path: str, record: dict) -> None:
    """Append one JSON line to the status file under an exclusive flock (the close releases it)."""
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + "\n")
        f.flush()


def read_history(path: str, task_id: str) -> list:
    """The attempts every pool recorded for the task, oldest first (the status lines of outcome ok, retry or failed).

    The file is read under a shared flock, so a line being appended is read whole or not at all. Lines that are not
    JSON objects are ignored."""
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return []
    with f:
        fcntl.flock(f, fcntl.LOCK_SH)
        text = f.read()
    rows = []
    for line in text.split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue                                  # blank, or not a line this module wrote
        if isinstance(row, dict) and row.get("id") == task_id and row.get("outcome") in ATTEMPT_OUTCOMES:
            rows.append(row)
    return rows


def current_attempts(rows: list) -> list:
    """The attempts that count against the budget: those after the last success, from the last first attempt on."""
    start = 0
    for index, row in enumerate(rows):
        if row["outcome"] == "ok":
            start = index + 1
        elif row.get("attempt") == 1:
            start = index
    return rows[start:]


def budget_spent(attempts: list, retries: int) -> bool:
    """True if the task has a `failed` attempt, or its attempts reach retries + 1 (the argument is current_attempts)."""
    return any(row["outcome"] == "failed" for row in attempts) or len(attempts) >= retries + 1


# ------------------------------------------------------------------ done files
def set_aside(path: str, attempt: int) -> str | None:
    """Rename the done file at path to <path>.failed-a<attempt> (a numbered variant if that name is taken).

    Returns the new name, or None if there was no file. Nothing is ever deleted or overwritten."""
    if not os.path.lexists(path):
        return None
    target, number = f"{path}.failed-a{attempt}", 1
    while os.path.lexists(target):
        number += 1
        target = f"{path}.failed-a{attempt}.{number}"
    os.rename(path, target)
    return target
