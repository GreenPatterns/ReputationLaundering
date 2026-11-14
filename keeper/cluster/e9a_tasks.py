"""E9a task list (Task 4.8): the self-term ablation runs of protocol Section 2.11, which also feed E8.

  python -m keeper.cluster.e9a_tasks --results-root ABS_REPO_ROOT --out TASKS.tsv [--force]

Writes the task file that keeper.cluster.taskpool reads, in the format of keeper.cluster.cppr_tasks (one task per
line, `id<TAB>done_path<TAB>log_path<TAB>command`), whose helpers it reuses. The pool runs each command with its
working directory set to a git worktree fixed at the run commit, so `--prereg` is the protocol's path inside that
worktree; the runs write by absolute path under --results-root, the main repository (never a git worktree).

30 tasks: per network (largest first by registered n_ref), seed in SEEDS_5 (5-9) and self_term in (off, on):
  id      e9a_{ds}_s{seed}_{off|on}
  result  results/laundering/e9a/<id>.json   (the done path)
  log     results/laundering/e9a/logs/<id>.log
  dump    results/dumps/e9a/<id>.npz         (git-ignored; keeper.laundering.e9a_runner names it after the result)
Exit status as cppr_tasks: 0 written; 1 exists without --force or unwritable; 2 usage error.
"""
import argparse
import os
import shlex
import sys

from keeper.cluster import protocol_settings as ps
from keeper.cluster.cppr_tasks import TaskRow, _checked_root, _largest_first, _quoted, render, write_task_file

RUNNER = "keeper.laundering.e9a_runner"
SELF_TERMS = ("off", "on")


def task_id(ds: str, seed: int, self_term: str) -> str:
    return f"e9a_{ds}_s{seed}_{self_term}"


def e9a_tasks(results_root: str) -> list[TaskRow]:
    """The 30 tasks of E9a, in task-file order."""
    root = _checked_root(results_root)
    folder = os.path.join(root, "results", "laundering", "e9a")
    rows = []
    for ds in _largest_first():
        for seed in ps.SEEDS_5:
            for self_term in SELF_TERMS:
                tid = task_id(ds, seed, self_term)
                done = os.path.join(folder, f"{tid}.json")
                command = " ".join([ps.PYTHON, "-m", RUNNER, "--ds", ds, "--seed", str(seed), "--self-term", self_term,
                                    "--epochs", str(ps.EPOCHS), "--prereg", ps.PREREG, "--out", shlex.quote(done),
                                    "--dump-dir", _quoted(root, "results", "dumps", "e9a")])
                rows.append(TaskRow(tid, done, os.path.join(folder, "logs", f"{tid}.log"), command))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.cluster.e9a_tasks",
                                 description="Write the task file of the E9a runs, for keeper.cluster.taskpool.")
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="absolute path of the main repository, not of a git worktree")
    ap.add_argument("--out", required=True, metavar="TASKS.tsv", help="the task file to write")
    ap.add_argument("--force", action="store_true", help="replace an existing task file")
    args = ap.parse_args(argv)
    try:
        rows = e9a_tasks(args.results_root)
    except ValueError as error:
        ap.error(str(error))
    try:
        write_task_file(args.out, render(rows), args.force)
    except FileExistsError:
        print(f"e9a_tasks: {args.out} exists; pass --force to replace it", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"e9a_tasks: cannot write {args.out}: {error.strerror}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} tasks (E9a) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
