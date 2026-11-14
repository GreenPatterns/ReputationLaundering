"""E7 task list (Task 4.7): KEEPER combined, docs/preregistration.md Sections 2.1, 2.9 and 6.

  python -m keeper.cluster.e7_tasks --results-root ABS_REPO_ROOT --out TASKS.tsv [--force]

Writes the task file that keeper.cluster.taskpool reads, in the format of keeper.cluster.cppr_tasks: one task per line,
`id<TAB>done_path<TAB>log_path<TAB>command`. Every value comes from keeper.cluster.protocol_settings. The pool runs each
command with its working directory set to a git worktree fixed at the run commit, so `--prereg` is the protocol's path
inside that worktree; the runs write by absolute path under `--results-root`, the main repository (never a worktree).

Runs, 8 per network and seed in SEEDS_5 and 120 in all, outputs results/cppr/e7/<id>.json (the done path), logs
results/cppr/e7/logs/<id>.log. Networks come largest first, seeds ascend; within a seed the CPPR runs come first.
  e7_cppr_g{0,1}_{ds}_s{seed}   the CPPR driver with --backbone rating_scem, capped BTP and --hold-test-snapshot
                                (Section 6), the arms and methods of protocol_settings.E7_CPPR[g]; g = 1 adds
                                --attack-edges 1 and --pool-eps EPS_C (random endorsers, ring, s = 3: the driver's
                                defaults). Target dumps go to results/cppr/e7/targets.
  e7_witness_{clean,g0,g1}_mb{on,off}_{ds}_s{seed}   keeper.cppr.witness_arm.

Exit status: 0 when the file is written; 1 when it cannot be written (it exists and --force is missing, or its folder
is missing); 2 for a usage error, such as a --results-root that is relative or a git worktree.
"""
import argparse
import os
import shlex
import sys

from keeper.cluster import protocol_settings as ps
from keeper.cluster.cppr_tasks import TaskRow, _checked_root, _command, _largest_first, render, write_task_file

WITNESS_DRIVER = "keeper.cppr.witness_arm"
FOLDER = ("results", "cppr", "e7")


def cppr_runs() -> list:
    """(ds, seed, g) of every registered E7 CPPR run, in task-file order."""
    return [(ds, seed, g) for ds in _largest_first() for seed in ps.SEEDS_5 for g in ps.E7_CPPR]


def witness_runs() -> list:
    """(ds, seed, stream, mb) of every registered witness-arm run, in task-file order."""
    return [(ds, seed, stream, mb) for ds in _largest_first() for seed in ps.SEEDS_5
            for stream in ps.E7_WITNESS_STREAMS for mb in ps.E7_MB]


def cppr_id(ds: str, seed: int, g: int) -> str:
    return f"e7_cppr_g{g}_{ds}_s{seed}"


def witness_id(ds: str, seed: int, stream: str, mb: str) -> str:
    return f"e7_witness_{stream}_mb{mb}_{ds}_s{seed}"


def _row(folder: str, task_id: str, command) -> TaskRow:
    done = os.path.join(folder, f"{task_id}.json")
    return TaskRow(task_id, done, os.path.join(folder, "logs", f"{task_id}.log"), command(done))


def _cppr_extra(folder: str, ds: str, g: int) -> list:
    arms, methods = ps.E7_CPPR[g]
    attack = ["--attack-edges", str(g), "--pool-eps", str(ps.EPS_CAPPED[ds])] if g else []
    return ["--backbone", ps.E7_BACKBONE, "--arms", arms, "--methods", methods, *attack,
            "--dump-targets", shlex.quote(os.path.join(folder, "targets"))]


def _witness_command(ds: str, seed: int, stream: str, mb: str, done: str) -> str:
    return " ".join([ps.PYTHON, "-m", WITNESS_DRIVER, "--ds", ds, "--seed", str(seed), "--stream", stream, "--mb", mb,
                     "--epochs", str(ps.EPOCHS), "--prereg", ps.PREREG, "--out", shlex.quote(done)])


def e7_tasks(results_root: str) -> list[TaskRow]:
    """The E7 tasks, in task-file order: per network and seed the two CPPR runs, then the six witness-arm runs."""
    folder = os.path.join(_checked_root(results_root), *FOLDER)
    rows = []
    for ds in _largest_first():
        for seed in ps.SEEDS_5:
            rows += [_row(folder, cppr_id(ds, seed, g),
                          lambda done, g=g: _command(ds, seed, done, ps.CAPPED, _cppr_extra(folder, ds, g)))
                     for g in ps.E7_CPPR]
            rows += [_row(folder, witness_id(ds, seed, stream, mb),
                          lambda done, stream=stream, mb=mb: _witness_command(ds, seed, stream, mb, done))
                     for stream in ps.E7_WITNESS_STREAMS for mb in ps.E7_MB]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m keeper.cluster.e7_tasks",
                                 description="Write the task file of the E7 runs, for keeper.cluster.taskpool.")
    ap.add_argument("--results-root", required=True, metavar="ABS_REPO_ROOT",
                    help="absolute path of the main repository, not of a git worktree")
    ap.add_argument("--out", required=True, metavar="TASKS.tsv", help="the task file to write")
    ap.add_argument("--force", action="store_true", help="replace an existing task file")
    args = ap.parse_args(argv)
    try:
        rows = e7_tasks(args.results_root)
    except ValueError as error:
        ap.error(str(error))
    try:
        write_task_file(args.out, render(rows), args.force)
    except FileExistsError:
        print(f"e7_tasks: {args.out} exists; pass --force to replace it", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"e7_tasks: cannot write {args.out}: {error.strerror}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} tasks (E7) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
