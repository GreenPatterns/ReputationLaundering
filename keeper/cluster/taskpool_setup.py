"""The task file and the checks before anything starts (Task 4.1a): what keeper.cluster.taskpool refuses to run with.

Task file. One task per line, four tab-separated fields: id, done_path, log_path, command. The command is the rest of
the line and may hold tabs. Blank lines and lines starting with `#` are ignored; every field is stripped, so CRLF
line endings are harmless. The file is checked as a whole: ids and done paths are unique, done and log paths are
absolute, no log path is a done path (a log would then pass for a result), id and command are not empty, and there
is at least one task. SetupError names every bad line, counting the physical lines of the file.

GPUs. The ids come from --gpus or, by default, from CUDA_VISIBLE_DEVICES; they are opaque strings and a repeated id
counts once. When both are set, --gpus must be a subset of CUDA_VISIBLE_DEVICES (an empty value counts as unset), so
that a task is never sent to a GPU the job does not own.

Folders. The status file's folder and the log and done folders of every task must be writable. They are checked
without creating anything (the nearest folder that exists must take new files), so that a typo in a path stops the
pool before the first task and not after hours.
"""
import dataclasses
import os

from keeper.cluster.taskpool_state import CLAIM_SUFFIX

MAX_FOLDERS_NAMED = 8                    # how many unwritable folders an error message lists


class SetupError(Exception):
    """The pool cannot start; the message says what to fix."""


@dataclasses.dataclass(frozen=True)
class Task:
    id: str
    done_path: str
    log_path: str
    command: str
    line: int                            # 1-based line number in the task file

    @property
    def claim_path(self) -> str:
        return self.done_path + CLAIM_SUFFIX


# ------------------------------------------------------------------ task file
def _field_problems(fields: list) -> list:
    """What is wrong with the fields of one task line (an empty list if nothing)."""
    if len(fields) != 4:
        return [f"expected 4 tab-separated fields (id, done_path, log_path, command), found {len(fields)}"]
    task_id, done_path, log_path, command = (field.strip() for field in fields)
    problems = [] if task_id else ["empty id"]
    for name, path in (("done_path", done_path), ("log_path", log_path)):
        if not os.path.isabs(path):
            problems.append(f"{name} {path!r} is not an absolute path")
    return problems + ([] if command else ["empty command"])


def _clashes(tasks: list) -> list:
    """(line, message) for a task whose id or done_path an earlier task has, and for a log_path that is a done_path."""
    problems, first_id, first_done = [], {}, {}
    for task in tasks:
        first_done.setdefault(task.done_path, task.line)
    for task in tasks:
        if task.id in first_id:
            problems.append((task.line, f"duplicate id {task.id!r} (first on line {first_id[task.id]})"))
        if first_done[task.done_path] != task.line:
            problems.append((task.line, f"same done_path as line {first_done[task.done_path]}: {task.done_path}"))
        if task.log_path in first_done:
            problems.append((task.line, f"log_path is the done_path of line {first_done[task.log_path]}"))
        first_id.setdefault(task.id, task.line)
    return problems


def parse_tasks(text: str, name: str = "task file") -> list:
    """The tasks of a task file's text, in file order. SetupError names every bad line (physical line numbers)."""
    tasks, problems = [], []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t", 3)
        found = _field_problems(fields)
        problems += [(number, problem) for problem in found]
        if not found:
            tasks.append(Task(*(field.strip() for field in fields), line=number))
    problems += _clashes(tasks)
    if problems:
        raise SetupError("\n".join(f"{name}: line {number}: {problem}" for number, problem in sorted(problems)))
    if not tasks:
        raise SetupError(f"{name}: no tasks (only blank and comment lines)")
    return tasks


def load_tasks(path: str) -> list:
    """The tasks of the task file at path; SetupError if it cannot be read or is bad."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as error:
        raise SetupError(f"cannot read the task file {path}: {error.strerror}") from None
    except UnicodeDecodeError:
        raise SetupError(f"{path} is not a UTF-8 text file") from None
    return parse_tasks(text, path)


# ------------------------------------------------------------------ GPUs, working directory, folders
def parse_gpus(spec) -> list:
    """The distinct GPU ids of a comma-separated list, in order; empty for None or a list without ids."""
    ids = (gpu.strip() for gpu in (spec or "").split(","))
    return list(dict.fromkeys(gpu for gpu in ids if gpu))


def resolve_gpus(option) -> tuple:
    """The GPU ids of --gpus, else of CUDA_VISIBLE_DEVICES. SetupError if there are none, or if --gpus names a GPU
    that a non-empty CUDA_VISIBLE_DEVICES does not offer."""
    visible_text = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = parse_gpus(visible_text)
    gpus = parse_gpus(option if option is not None else visible_text)
    if not gpus:
        source = "--gpus" if option is not None else "CUDA_VISIBLE_DEVICES"
        raise SetupError(f"no GPU to run on: {source} lists none (pass --gpus 0,1,... or start the pool in an "
                         "allocation that sets CUDA_VISIBLE_DEVICES)")
    outside = [gpu for gpu in gpus if option is not None and visible and gpu not in visible]
    if outside:
        raise SetupError(f"--gpus names GPU {', '.join(outside)}, which this job does not see "
                         f"(CUDA_VISIBLE_DEVICES={visible_text.strip()})")
    return tuple(gpus)


def resolve_cwd(option: str) -> str:
    cwd = os.path.abspath(option)
    if not os.path.isdir(cwd):
        raise SetupError(f"--cwd {cwd} is not a directory")
    return cwd


def _takes_new_files(folder: str) -> bool:
    """True if new files can be made in folder, or in the nearest folder above it that exists."""
    while not os.path.exists(folder):
        folder = os.path.dirname(folder)
    return os.path.isdir(folder) and os.access(folder, os.W_OK | os.X_OK)


def check_writable(tasks: list, status_path: str) -> None:
    """SetupError unless the status file and the log and done folders of every task can be written (nothing is made)."""
    folders = {os.path.dirname(os.path.abspath(status_path))}
    for task in tasks:
        folders.update((os.path.dirname(task.done_path), os.path.dirname(task.log_path)))
    bad = sorted(folder for folder in folders if not _takes_new_files(folder))
    if os.path.exists(status_path) and not os.access(status_path, os.W_OK):
        bad.append(status_path)
    if bad:
        more = f" and {len(bad) - MAX_FOLDERS_NAMED} more" if len(bad) > MAX_FOLDERS_NAMED else ""
        raise SetupError(f"cannot write to {', '.join(bad[:MAX_FOLDERS_NAMED])}{more}")
