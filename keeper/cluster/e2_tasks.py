"""E2 task list (Task 4.5): the SCEM end-to-end silent-sender runs of docs/preregistration.md, Sections 2.1 and 2.4.

  python -m keeper.cluster.e2_tasks --results-root ABS_REPO_ROOT --out TASKS.tsv [--force]

Writes the task file that keeper.cluster.taskpool reads, in the format of keeper.cluster.cppr_tasks: one task per line,
`id<TAB>done_path<TAB>log_path<TAB>command`. Every value comes from keeper.cluster.protocol_settings. The pool runs each
command with its working directory set to a git worktree fixed at the run commit, so `--prereg` is the protocol's path
inside that worktree; the runs write by absolute path under `--results-root`, the main repository (never a worktree).

Runs, 105 per network and 315 in all: per network and seed in SEEDS_5, every stream (clean, lag1, lag3) with every
model (s3r, scem, scem_phi05, scem_nohaz, scem_n1); per seed in SEEDS_E2_H6 outside SEEDS_5 (10-14), the clean twin and
lag 1 with the H6 models (s3r, scem, scem_nohaz). Task id e2_{model}_{stream}_{ds}_s{seed}; output
results/scem/e2/<id>.json (the done path); log results/scem/e2/logs/<id>.log. Networks come largest first.

Exit status: 0 when the file is written; 1 when it cannot be written (it exists and --force is missing, or its folder
is missing); 2 for a usage error, such as a --results-root that is relative or a git worktree.
"""
import argparse
import os
import shlex
import sys

from keeper.cluster import protocol_settings as ps
from keeper.cluster.cppr_tasks import TaskRow, _checked_root, _largest_first, render, write_task_file

DRIVER = "keeper.scem.e2_driver"
FOLDER = ("results", "scem", "e2")


def e2_runs() -> list:
    """(ds, seed, stream, model) of every registered E2 run, in task-file order."""
    runs = []
    for ds in _largest_first():
        for seed in ps.SEEDS_E2_H6:
            full = seed in ps.SEEDS_5
            for stream in ps.E2_STREAMS if full else ps.E2_H6_STREAMS:
                for model in ps.E2_MODELS if full else ps.E2_H6_MODELS:
                    runs.append((ds, seed, stream, model))
    return runs


def task_id(ds: str, seed: int, stream: str, model: str) -> str:
    return f"e2_{model}_{stream}_{ds}_s{seed}"


def e2_tasks(results_root: str) -> list[TaskRow]:
    """The E2 tasks, in task-file order."""
    folder = os.path.join(_checked_root(results_root), *FOLDER)
    rows = []
    for ds, seed, stream, model in e2_runs():
        tid = task_id(ds, seed, stream, model)
        done = os.path.join(folder, f"{tid}.json")
        command = " ".join([ps.PYTHON, "-m", DRIVER, "--ds", ds, "--seed", str(seed), "--stream", stream,
                            "--model", model, "--epochs", str(ps.EPOCHS), "--prereg", ps.PREREG,
                            "--out", shlex.quote(done)])
        rows.append(TaskRow(tid, done, os.path.join(folder, "logs", f"{tid}.log"), command))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.cluster.e2_tasks",
                                 description="Write the task file of the E2 runs, for keeper.cluster.taskpool.")
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="absolute path of the main repository, not of a git worktree")
    ap.add_argument("--out", required=True, metavar="TASKS.tsv", help="the task file to write")
    ap.add_argument("--force", action="store_true", help="replace an existing task file")
    args = ap.parse_args(argv)
    try:
        rows = e2_tasks(args.results_root)
    except ValueError as error:
        ap.error(str(error))
    try:
        write_task_file(args.out, render(rows), args.force)
    except FileExistsError:
        print(f"e2_tasks: {args.out} exists; pass --force to replace it", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"e2_tasks: cannot write {args.out}: {error.strerror}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} tasks (E2) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
